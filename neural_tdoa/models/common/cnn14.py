from neural_tdoa.models.common.model_utilities import MelSpectrogramArray
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torchlibrosa.augmentation import SpecAugmentation

from neural_tdoa.models.settings import (
    PRETRAINED_MODEL_PATH,
    N_MELS, N_FFT, HOP_LENGTH
)
from datasets.settings import N_MICS, SR

CNN14_PRETRAINED_MODEL_PATH = Path(PRETRAINED_MODEL_PATH) / "Cnn14_mAP=0.431.pth"  # noqa
CNN14_PRETRAINED_MODEL_PATH = Path(PRETRAINED_MODEL_PATH) / "multichannel_cnn14.pth"  # noqa

CNN14_OUTPUT_FEATURE_SIZE = 2048
AUDIOSET_CLASSES_NUM = 527


class Cnn14(nn.Module):
    def __init__(
            self, sr=SR,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS):

        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.mel_spectrogram = MelSpectrogramArray(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=N_MICS, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, 
                                     out_channels=CNN14_OUTPUT_FEATURE_SIZE)

        self.fc1 = nn.Linear(
            CNN14_OUTPUT_FEATURE_SIZE, CNN14_OUTPUT_FEATURE_SIZE, bias=True)
        self.fc_audioset = nn.Linear(
            CNN14_OUTPUT_FEATURE_SIZE, AUDIOSET_CLASSES_NUM, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.mel_spectrogram(input)
        # (batch_size, mic_channels, freq_bins, time_steps)
        # x = x.transpose(1, 3)
        #x = self.bn0(x)
        # x = x.transpose(1, 3)

        # if self.training:
        #     x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        return embedding
        #clipwise_output = torch.sigmoid(self.fc_audioset(x))
        # output_dict = {
        #     'clipwise_output': clipwise_output,
        #     'embedding': embedding
        # }

        # return output_dict


def load_pretrained_cnn14(
        pretrained_checkpoint_path=CNN14_PRETRAINED_MODEL_PATH,
        trainable=True):

    checkpoint = torch.load(
        pretrained_checkpoint_path, map_location=torch.device('cpu'))

    model = Cnn14()
    model.load_state_dict(checkpoint['model'])

    if not trainable:
        for param in model.parameters():
            param.requires_grad = False

    return model


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x
