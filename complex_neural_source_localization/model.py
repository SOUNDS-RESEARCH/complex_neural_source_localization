# Credits to Yin Cao et al:
# https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization/blob/master/models/CRNNs.py

import torch
import torch.nn as nn

from complex_neural_source_localization.feature_extractors import DecoupledStftArray, StftArray

from .utils.model_utilities import ConvBlock, init_gru, init_layer

DEFAULT_CONV_CONFIG = [
    {"type": "real_single", "n_channels": 64},
    {"type": "real_double", "n_channels": 128},
    {"type": "real_double", "n_channels": 256},
    {"type": "real_double", "n_channels": 512},
]

DEFAULT_STFT_CONFIG = {"n_fft": 1024, "hop_length":480}


class Crnn10(nn.Module):
    def __init__(self, output_type="scalar", n_input_channels=4,
                 pool_type="avg", pool_size=(2,2),
                 complex_to_real_function="concatenate",
                 conv_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 init_conv_layers=False,
                 last_layer_dropout_rate=0.5):
        
        super().__init__()

        self.n_input_channels = n_input_channels
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.output_type = output_type
        self.complex_to_real_function = complex_to_real_function

        if "complex" in conv_config[0]["type"]:
            self.feature_extractor = StftArray(stft_config)
        else:
            self.feature_extractor = DecoupledStftArray(stft_config)

        self.conv_blocks = self._init_conv_blocks(conv_config, init_conv_layers=init_conv_layers)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, batch_first=True, bidirectional=True)

        if last_layer_dropout_rate > 0:
            self.azimuth_fc = nn.Sequential(
                nn.Linear(512, 2, bias=True), # 2 cartesian dimensions
                nn.Dropout(last_layer_dropout_rate)
            )
        else:
            self.azimuth_fc = nn.Linear(512, 2, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        init_gru(self.gru)
        init_layer(self.azimuth_fc)

    def _init_conv_blocks(self, conv_config, init_conv_layers):
        
        conv_blocks = [
            ConvBlock(self.n_input_channels, conv_config[0]["n_channels"],
                          block_type=conv_config[0]["type"],
                          dropout_rate=conv_config[0]["dropout_rate"])
        ]

        for config in conv_config[1:]:
            last_layer = conv_blocks[-1]
            in_channels = last_layer.out_channels
            if last_layer.is_real == False and config["type"] != "complex": # complex number will be unpacked into 2x channels
                in_channels *= 2
            conv_blocks.append(
                ConvBlock(in_channels, config["n_channels"],
                          block_type=config["type"], init=init_conv_layers,
                          dropout_rate=config["dropout_rate"])
            )
        
        return nn.ModuleList(conv_blocks)
        
    def forward(self, x):
        """input: (batch_size, mic_channels, temporal_time_steps)"""

        x = self.feature_extractor(x)
        """(batch_size, mic_channels, n_freqs, time_steps)"""
        x = x.transpose(2, 3)
        """(batch_size, mic_channels, n_freqs)"""
        

        for conv_block in self.conv_blocks:
            if x.is_complex() and conv_block.is_real:
                x = _to_real(x, mode=self.complex_to_real_function)
            x = conv_block(x)
        
        if x.is_complex():
            x = _to_real(x, mode=self.complex_to_real_function)
        """(batch_size, feature_maps, time_steps, n_freqs)"""

        x = torch.mean(x, dim=3)
        """(batch_size, feature_maps, time_steps)"""

        x = x.transpose(1,2)
        """ (batch_size, time_steps, feature_maps):"""

        (x, _) = self.gru(x)
        
        x = self.azimuth_fc(x)

        if self.output_type == "scalar":
        # """(batch_size, time_steps, class_num)"""
            if self.pool_type == "avg":
                x = torch.mean(x, dim=1)
            elif self.pool_type == "max":
                (x, _) = torch.max(x, dim=1)

        return x


def _to_real(x, mode="concatenate"):
    real_part = x.real
    imag_part = x.imag
    
    if mode == "concatenate":
        x = torch.cat([real_part, imag_part], axis=1)
    elif mode == "magnitude":
        x = x.abs()
    elif mode == "phase":
        x = x.angle()

    return x
