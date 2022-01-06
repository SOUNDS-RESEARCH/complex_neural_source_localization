# Credits to Yin Cao et al:
# https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization/blob/master/models/CRNNs.py

import torch
import torch.nn as nn

from neural_tdoa.feature_extractors import DecoupledStftArray, StftArray

from .utils.model_utilities import ComplexConvBlock, ConvBlock, init_gru, init_layer


class Crnn10(nn.Module):
    def __init__(self, output_type="scalar", n_input_channels=4, pool_type='avg', pool_size=(2,2),
                        complex_to_real_function="concatenate"):
        
        super().__init__()

        self.pool_type = pool_type
        self.pool_size = pool_size
        self.output_type = output_type
        self.complex_to_real_function = complex_to_real_function

        #self.feature_extractor = DecoupledStftArray({"n_fft": 1024, "hop_length":480})
        self.feature_extractor = StftArray({"n_fft": 1024, "hop_length":480})
        self.conv_block1 = ComplexConvBlock(in_channels=n_input_channels, out_channels=64)
        #self.conv_block1 = ConvBlock(in_channels=n_input_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, batch_first=True, bidirectional=True)

        self.azimuth_fc = nn.Linear(512, 2, bias=True) # 2 cartesian dimensions

        self.init_weights()

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.azimuth_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, temporal_time_steps)'''

        x = self.feature_extractor(x)
        '''(batch_size, mic_channels, n_freqs, time_steps)'''
        x = x.transpose(2, 3)
        '''(batch_size, mic_channels, n_freqs)'''
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = _to_real(x, mode=self.complex_to_real_function)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, n_freqs)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)
        
        x = self.azimuth_fc(x)

        if self.output_type == "scalar":
        # '''(batch_size, time_steps, class_num)'''
            if self.pool_type == 'avg':
                x = torch.mean(x, dim=1)
            elif self.pool_type == 'max':
                (x, _) = torch.max(x, dim=1)

        return x


def _to_real(conv_results, mode="concatenate"):
    real_part = conv_results.real
    imag_part = conv_results.imag
    
    if mode == "concatenate":
        x = torch.cat([real_part, imag_part], axis=1)
    elif mode == "magnitude":
        x = conv_results.abs()
    elif mode == "phase":
        x = conv_results.angle()

    return x
