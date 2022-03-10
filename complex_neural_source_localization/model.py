from typing import Tuple
import torch
import torch.nn as nn

from complex_neural_source_localization.feature_extractors import DecoupledStftArray, StftArray

from .utils.model_utilities import ConvBlock, init_gru, init_layer

DEFAULT_CONV_CONFIG = [
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
]

DEFAULT_STFT_CONFIG = {"n_fft": 1024}


class DOACNet(nn.Module):
    def __init__(self, output_type="scalar", n_input_channels=4, n_sources=2,
                 pool_type="avg", pool_size=(2,2),
                 complex_to_real_function="concatenate",
                 conv_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 init_conv_layers=False,
                 last_layer_dropout_rate=0.5,
                 store_feature_maps=False):
        
        super().__init__()

        # 1. Store configuration
        self.n_input_channels = n_input_channels
        self.n_sources = n_sources
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.output_type = output_type
        self.complex_to_real_function = complex_to_real_function
        self.max_filters = conv_config[-1]["n_channels"]

        # 2. Create feature extractor
        self.feature_extractor = self._create_feature_extractor(stft_config, conv_config[0]["type"])

        # 3. Create convolutional blocks
        self.conv_blocks = self._create_conv_blocks(conv_config, init_conv_layers=init_conv_layers)

        # 4. Create recurrent block
        self.rnn = self._create_rnn_block(self.max_filters)

        # 5. Create linear block
        self.azimuth_fc = self._create_linear_block(n_sources, last_layer_dropout_rate)

        self._init_weights()
        
        if store_feature_maps:
            self.track_feature_maps()
    
    def forward(self, x):
        # input: (batch_size, mic_channels, time_steps)

        x = self.feature_extractor(x)
        # (batch_size, mic_channels, n_freqs, stft_time_steps)
        x = x.transpose(2, 3)
        # (batch_size, mic_channels, stft_time_steps, n_freqs)
        
        for conv_block in self.conv_blocks:
            if x.is_complex() and conv_block.is_real:
                x = _to_real(x, mode=self.complex_to_real_function)
            x = conv_block(x)
        
        if x.is_complex():
            x = _to_real(x, mode=self.complex_to_real_function)
        # (batch_size, feature_maps, time_steps, n_freqs)

        # Average across all frequencies
        x = torch.mean(x, dim=3)
        # (batch_size, feature_maps, time_steps)

        x = x.transpose(1,2)
        # (batch_size, time_steps, feature_maps):

        (x, _) = self.rnn(x)
        # (batch_size, time_steps, feature_maps):

        x = self.azimuth_fc(x)
        # (batch_size, time_steps, class_num)

        # Average across all time steps
        if self.output_type == "scalar":
            if self.pool_type == "avg":
                x = torch.mean(x, dim=1)
            elif self.pool_type == "max":
                (x, _) = torch.max(x, dim=1)

        return x

    def _create_feature_extractor(self, stft_config, first_conv_layer_type):
        if "complex" in first_conv_layer_type:
            return StftArray(stft_config)
        else:
            return DecoupledStftArray(stft_config)

    def _create_conv_blocks(self, conv_config, init_conv_layers):
        
        conv_blocks = [
            ConvBlock(self.n_input_channels, conv_config[0]["n_channels"],
                      block_type=conv_config[0]["type"],
                      dropout_rate=conv_config[0]["dropout_rate"])
        ]

        for i, config in enumerate(conv_config[1:]):
            last_layer = conv_blocks[-1]
            in_channels = last_layer.out_channels
            if last_layer.is_real == False and "complex" not in config["type"]:
                # complex convolutions are performed using 2 convolutions of half the filters
                in_channels *= 2
            conv_blocks.append(
                ConvBlock(in_channels, config["n_channels"],
                          block_type=config["type"], init=init_conv_layers,
                          dropout_rate=config["dropout_rate"])
            )
        
        return nn.ModuleList(conv_blocks)
        
    def _create_rnn_block(self, input_size):
            return nn.GRU(input_size=input_size,
                          hidden_size=input_size//2, 
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

    def _create_linear_block(self, n_sources, last_layer_dropout_rate):   
        n_last_layer = 2*n_sources  # 2 cartesian dimensions for each source

        if last_layer_dropout_rate > 0:
            return nn.Sequential(
                nn.Linear(self.max_filters, n_last_layer, bias=True),
                nn.Dropout(last_layer_dropout_rate)
            )
        else:
            return nn.Linear(self.max_filters, n_last_layer, bias=True)
    
    def _init_weights(self):
        init_gru(self.rnn)
        init_layer(self.azimuth_fc)
    
    def track_feature_maps(self):
        "Make all the intermediate layers accessible through the 'feature_maps' dictionary"

        self.feature_maps = {}

        hook_fn = self._create_hook_fn("stft")
        self.feature_extractor.register_forward_hook(hook_fn)
        
        for i, conv_layer in enumerate(self.conv_blocks):
            hook_fn = self._create_hook_fn(f"conv_{i}")
            conv_layer.register_forward_hook(hook_fn)
        
        hook_fn = self._create_hook_fn("rnn")
        self.rnn.register_forward_hook(hook_fn)

        hook_fn = self._create_hook_fn("azimuth_fc")
        self.azimuth_fc.register_forward_hook(hook_fn)

    def _create_hook_fn(self, layer_id):
        def fn(_, __, output):
            if type(output) == tuple:
                output = output[0]
            self.feature_maps[layer_id] = output.detach().cpu() #.cpu().detach()
        return fn


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
