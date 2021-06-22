import torch
from torch.nn import Module, ModuleList

from yoho.model.common.cnn14 import (
    ConvBlock, load_pretrained_cnn14,
    CNN14_PRETRAINED_MODEL_PATH,
    CNN14_OUTPUT_FEATURE_SIZE
)
from datasets.settings import N_MICS
from yoho.settings import IS_BASE_TRAINABLE


class MultichannelCnn14(Module):
    def __init__(
            self, num_channels=N_MICS,
            pretrained_checkpoint_path=CNN14_PRETRAINED_MODEL_PATH,
            is_base_trainable=IS_BASE_TRAINABLE):

        super().__init__()

        self.num_channels = num_channels
        self.base = load_pretrained_cnn14(
            pretrained_checkpoint_path,
            trainable=is_base_trainable
        )

        self.fc_output = torch.nn.Linear(
            CNN14_OUTPUT_FEATURE_SIZE, 
            CNN14_OUTPUT_FEATURE_SIZE//num_channels,
            bias=True
        )

    def forward(self, x):
        """compute events on all channels

        Args:
            x (torch.Tensor): Multichannel audio signal
        """

        def get_channel_result(x, n_channel):
            x = self.base(x[:, n_channel, :])
            x["embedding"] = self.fc_output(x["embedding"])
            return x

        result_dicts = [
            get_channel_result(x, channel)
            for channel in range(self.num_channels)
        ]

        embeddings = [r["embedding"] for r in result_dicts]
        predictions = [r["clipwise_output"] for r in result_dicts]

        return {
            "embedding": torch.stack(embeddings, axis=1),
            "predictions": torch.stack(predictions, axis=1)
        }
