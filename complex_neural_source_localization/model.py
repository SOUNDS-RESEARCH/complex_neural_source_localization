import torch
import torch.nn as nn

from complex_neural_source_localization.feature_extractors import (
    CrossSpectra, DecoupledStftArray, StftArray
)
from complex_neural_source_localization.utils.conv_block import ConvBlock

from complex_neural_source_localization.utils.complexPyTorch.complexLayers import (
    ComplexGRU, ComplexLinear
)

DEFAULT_CONV_CONFIG = [
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
    {"type": "complex_single", "n_channels": 64, "dropout_rate":0},
]

DEFAULT_STFT_CONFIG = {"n_fft": 1024}


class DOACNet(nn.Module):
    def __init__(self, output_type="scalar", n_input_channels=4, n_sources=2,
                 pool_type="avg", pool_size=(1,2),
                 feature_type="stft",
                 conv_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 init_conv_layers=False,
                 last_layer_dropout_rate=0.5,
                 activation="relu",
                 complex_to_real_mode="amp_phase"):
        
        super().__init__()

        # 1. Store configuration
        if feature_type == "stft":
            self.n_input_channels = n_input_channels
        elif feature_type == "cross_spectra":
            self.n_input_channels = sum(range(n_input_channels + 1))
        self.n_sources = n_sources
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.output_type = output_type
        self.complex_to_real_mode = complex_to_real_mode
        self.activation = activation
        self.max_filters = conv_config[-1]["n_channels"]

        # 2. Create feature extractor
        self.feature_extractor = self._create_feature_extractor(feature_type, stft_config, conv_config[0]["type"])

        # 3. Create convolutional blocks
        self.conv_blocks = self._create_conv_blocks(conv_config, init_conv_layers=init_conv_layers)

        # 4. Create recurrent block
        self.rnn = self._create_rnn_block(self.max_filters)

        # 5. Create linear block
        self.azimuth_fc = self._create_linear_block(n_sources, last_layer_dropout_rate)
    
    def forward(self, x):
        # input: (batch_size, mic_channels, time_steps)

        x = self.feature_extractor(x)
        # (batch_size, mic_channels, n_freqs, stft_time_steps)
        x = x.transpose(2, 3)
        # (batch_size, mic_channels, stft_time_steps, n_freqs)
        
        for conv_block in self.conv_blocks:
            if x.is_complex() and conv_block.is_real:
                x = complex_to_real(x, mode=self.complex_to_real_mode)
            x = conv_block(x)
        
        # if x.is_complex():
        #     x = complex_to_real(x, mode=self.complex_to_real_mode)
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

        if x.is_complex():
            x = complex_to_real(x, mode="real_imag")

        return x

    def _create_feature_extractor(self, feature_type, stft_config, first_conv_layer_type):
        if feature_type == "cross_spectra":
            return CrossSpectra(stft_config)
        elif feature_type == "stft":  
            if "complex" in first_conv_layer_type:
                return StftArray(stft_config)
            else:
                return DecoupledStftArray(stft_config)

    def _create_conv_blocks(self, conv_config, init_conv_layers):
        
        conv_blocks = [
            ConvBlock(self.n_input_channels, conv_config[0]["n_channels"],
                      block_type=conv_config[0]["type"],
                      dropout_rate=conv_config[0]["dropout_rate"],
                      pool_size=self.pool_size),
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
                          dropout_rate=config["dropout_rate"],
                          pool_size=self.pool_size,
                          activation=self.activation)
            )
        
        return nn.ModuleList(conv_blocks)
        
    def _create_rnn_block(self, input_size):
        return ComplexGRU(input_size=input_size//2,
                          hidden_size=input_size//2) # TODO: Add bidirectional 

        # return nn.GRU(input_size=input_size,
        #                 hidden_size=input_size//2, 
        #                 num_layers=1,
        #                 batch_first=True,
        #                 bidirectional=True)

    def _create_linear_block(self, n_sources, last_layer_dropout_rate):   
        # n_last_layer = 2*n_sources  # 2 cartesian dimensions for each source

        # if last_layer_dropout_rate > 0:
        #     return nn.Sequential(
        #         nn.Linear(self.max_filters, n_last_layer, bias=True),
        #         nn.Dropout(last_layer_dropout_rate)
        #     )
        # else:
        #     return nn.Linear(self.max_filters, n_last_layer, bias=True)

        return ComplexLinear(self.max_filters//2, n_sources)
    
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


def complex_to_real(x, mode="real_imag"):
    real_part = x.real
    imag_part = x.imag
    
    if mode == "real_imag":
        x = torch.cat([real_part, imag_part], axis=1)
    elif mode == "magnitude":
        x = x.abs()
    elif mode == "phase":
        x = x.angle()
    elif mode == "amp_phase":
        x = torch.cat([x.abs(), x.angle()], axis=1)

    return x
