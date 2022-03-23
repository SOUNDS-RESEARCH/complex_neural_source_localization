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
                 pool_type="avg", pool_size=(1,2), kernel_size=(2, 2),
                 feature_type="stft",
                 conv_layers_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 init_conv_layers=False,
                 fc_layer_dropout_rate=0.5,
                 activation="relu",
                 complex_to_real_mode="real_imag",
                 use_complex_rnn=False,
                 **kwargs):
        
        super().__init__()

        # 1. Store configuration
        self.output_type = output_type
        self.n_input_channels = n_input_channels
        self.n_sources = n_sources
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.complex_to_real_mode = complex_to_real_mode
        self.activation = activation
        self.max_filters = conv_layers_config[-1]["n_channels"]
        self.is_rnn_complex = use_complex_rnn

        # 2. Create feature extractor
        self.feature_extractor = self._create_feature_extractor(feature_type, stft_config, conv_layers_config[0]["type"])

        # 3. Create convolutional blocks
        self.conv_blocks = self._create_conv_blocks(conv_layers_config, init_conv_layers=init_conv_layers)

        # 4. Create recurrent block
        self.rnn = self._create_rnn_block()

        # 5. Create linear block
        self.azimuth_fc = self._create_linear_block(n_sources, fc_layer_dropout_rate)
    
    def forward(self, x):
        # input: (batch_size, mic_channels, time_steps)

        # 1. Extract STFT of signals
        x = self.feature_extractor(x)
        # (batch_size, mic_channels, n_freqs, stft_time_steps)
        x = x.transpose(2, 3)
        # (batch_size, mic_channels, stft_time_steps, n_freqs)
        
        # 2. Extract features using convolutional layers
        for conv_block in self.conv_blocks:
            if x.is_complex() and conv_block.is_real:
                x = complex_to_real(x, mode=self.complex_to_real_mode)
            x = conv_block(x)
        # (batch_size, feature_maps, time_steps, n_freqs)

        # 3. Average across all frequencies
        x = torch.mean(x, dim=3)
        # (batch_size, feature_maps, time_steps)

        # Preprocessing for RNN
        if x.is_complex() and not self.is_rnn_complex:
            x = complex_to_real(x, mode=self.complex_to_real_mode)
        x = x.transpose(1,2)
        # (batch_size, time_steps, feature_maps):
        
        # 4. Use features as input to RNN
        (x, _) = self.rnn(x)
        # (batch_size, time_steps, feature_maps):

        # 5. Fully connected layer
        x = self.azimuth_fc(x)
        # (batch_size, time_steps, class_num)

        # Average across all time steps
        if self.output_type == "scalar":
            if self.pool_type == "avg":
                x = torch.mean(x, dim=1)
            elif self.pool_type == "max":
                (x, _) = torch.max(x, dim=1)
        elif self.output_type == "vector":
            # The network will predict one value per time step
            pass

        if x.is_complex():
            x = complex_to_real(x)
        return x

    def _create_feature_extractor(self, feature_type, stft_config, first_conv_layer_type):
        if feature_type == "stft":
            self.n_input_channels = self.n_input_channels
        elif feature_type == "cross_spectra":
            self.n_input_channels = sum(range(self.n_input_channels + 1))
        
        if feature_type == "cross_spectra":
            return CrossSpectra(stft_config)
        elif feature_type == "stft":  
            if "complex" in first_conv_layer_type:
                return StftArray(stft_config)
            else:
                return DecoupledStftArray(stft_config)

    def _create_conv_blocks(self, conv_layers_config, init_conv_layers):
        
        conv_blocks = [
            ConvBlock(self.n_input_channels, conv_layers_config[0]["n_channels"],
                      block_type=conv_layers_config[0]["type"],
                      dropout_rate=conv_layers_config[0]["dropout_rate"],
                      pool_size=self.pool_size,
                      activation=self.activation,
                      kernel_size=self.kernel_size),
        ]

        for i, config in enumerate(conv_layers_config[1:]):
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
                          activation=self.activation,
                          kernel_size=self.kernel_size)
            )
        
        return nn.ModuleList(conv_blocks)
        
    def _create_rnn_block(self):

        if self.is_rnn_complex:
            return ComplexGRU(input_size=self.max_filters//2,
                            hidden_size=self.max_filters//4,
                            batch_first=True, bidirectional=True)
        else:
            return nn.GRU(input_size=self.max_filters,
                          hidden_size=self.max_filters//2,
                          batch_first=True, bidirectional=True)

    def _create_linear_block(self, n_sources, fc_layer_dropout_rate):
        if self.is_rnn_complex:
            # TODO: Use dropout on complex linear block
            return ComplexLinear(self.max_filters//2, n_sources)
        else:
            n_last_layer = 2*n_sources  # 2 cartesian dimensions for each source            
            if fc_layer_dropout_rate > 0:
                return nn.Sequential(
                    nn.Linear(self.max_filters, n_last_layer, bias=True),
                    nn.Dropout(fc_layer_dropout_rate)
                )
            else:
                return nn.Linear(self.max_filters, n_last_layer, bias=True)
    
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


def complex_to_real(x, mode="real_imag", axis=1):
    if mode == "real_imag":
        x = torch.cat([x.real, x.imag], axis=axis)
    elif mode == "magnitude":
        x = x.abs()
    elif mode == "phase":
        x = x.angle()
    elif mode == "amp_phase":
        x = torch.cat([x.abs(), x.angle()], axis=axis)
    else:
        raise ValueError(f"Invalid complex mode :{mode}")

    return x
