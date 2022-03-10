import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from pathlib import Path
from torchvision.utils import make_grid

from complex_neural_source_localization.utils.model_utilities import (
    get_all_layers, ConvBlock
)


class ConvolutionalFeatureMapLogger:
    def __init__(self, model, trainer):
        # 1. Find convolutional layers in the model 
        self.conv_layers = get_all_layers(model, [ConvBlock])
        # 2. Variable to store the output feature maps produced in a forward pass
        self.feature_maps = {}
        # 3. Create a forward hook to fill the variable above at every pass

        for layer_id, layer in self.conv_layers.items():
            fn = self._create_hook(layer_id)
            layer.register_forward_hook(fn)

        self.trainer = trainer

    def log(self):
        n_epoch = self.trainer.current_epoch
        for layer, feature_maps in self.feature_maps.items():
            batch_sample_idx = 0 # Always select first example on batch
            feature_maps = feature_maps[batch_sample_idx]

            # Transform grayscale to RGB image
            # Make R and G channels 0 so we get a nice blue picture
            # Transpose time and frequency channels to get the format 
            # B x C x H x W required by torchvision's "make_grid" function
            feature_maps = feature_maps.unsqueeze(1).repeat([1, 3, 1, 1])
            feature_maps[:, 0:2, :, :] = 0

            if feature_maps.dtype == torch.complex64:
                feature_maps_mag = feature_maps.abs()
                feature_maps_phase = feature_maps.angle()
                # TODO: Phase unwrapping

                feature_maps_mag = make_grid(feature_maps_mag, normalize=True, padding=5)
                self.trainer.logger.experiment.add_image(f"{layer}.mag.epoch{n_epoch}", feature_maps_mag)
                feature_maps_phase = make_grid(feature_maps_phase, normalize=True, padding=5)
                self.trainer.logger.experiment.add_image(f"{layer}.phase.epoch{n_epoch}", feature_maps_phase)
            else:
                feature_maps = make_grid(feature_maps, normalize=True, padding=5)
                
                self.trainer.logger.experiment.add_image(f"{layer}.epoch{n_epoch}", feature_maps)


def plot_multichannel_spectrogram(multichannel_spectrogram, unwrap=True, mode="column",
                                  axs=None, figsize=(10, 5), output_path=None, close=True):
    
    num_channels, num_freq_bins, num_time_steps = multichannel_spectrogram.shape
    
    if axs is None:
        if mode == "column":
            n_rows, n_cols = (2, num_channels)
        elif mode == "row":
            n_rows, n_cols = (num_channels, 2)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    # If torch.Tensor, move to cpu and convert to numpy
    if isinstance(multichannel_spectrogram, torch.Tensor):
        multichannel_spectrogram = multichannel_spectrogram.cpu().detach().numpy()
    
    # Plot spectrograms for all channels
    for n_channel in range(num_channels):
        if mode == "row":
            channel_axs = (axs[n_channel][0], axs[n_channel][1])
        elif mode == "column":
            channel_axs = (axs[0][n_channel], axs[1][n_channel])
        else:
            raise ValueError("Allowed modes are 'row' and 'column'")

        plot_spectrogram(multichannel_spectrogram[n_channel], axs=channel_axs)

    if output_path is not None:
        plt.savefig(output_path)
        if close:
            plt.close()

    return axs


def plot_spectrogram(spectrogram, unwrap=True, mode="column", figsize=(10, 5), axs=None, output_path=None, close=True):
    if axs is None:
        if mode == "column":
            n_rows, n_cols = (2, 1)
        elif mode == "row":
            n_rows, n_cols = (1, 2)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    # If torch.Tensor, move to cpu and convert to numpy
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().detach().numpy()

    # Extract magnitude and phase, then unwrap phase
    spectrogram_mag_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    spectrogram_phase = np.angle(spectrogram)
    if unwrap:
        spectrogram_phase = np.unwrap(spectrogram_phase, axis=0)
    
    librosa.display.specshow(spectrogram_mag_db, ax=axs[0])
    librosa.display.specshow(spectrogram_phase, ax=axs[1])

    if output_path is not None:
        plt.savefig(output_path)
        if close:
            plt.close()
    
    return axs


def plot_model_output(feature_maps, metadata=None, unwrap=True,
                      batch_start_idx=0, output_dir_path=None, close_after_saving=True):
    
    output_dir_path = Path(output_dir_path)
    os.makedirs(output_dir_path, exist_ok=True)
    
    batch_size = feature_maps["stft"].shape[0]
    
    
    stft_output = feature_maps["stft"]
    conv_output = {
        feature_name: feature_map.transpose(2, 3)
        for feature_name, feature_map in feature_maps.items()
        if "conv" in feature_name
    }
    rnn_output = feature_maps["rnn"][0].transpose(1, 2)

    for i in range(batch_size):
        sample_idx = batch_size + batch_start_idx
        multichannel_spectrogram_filename = output_dir_path / f"{sample_idx}_multichannel_stft.png"

        plot_multichannel_spectrogram(
            stft_output[i], unwrap=unwrap,
            output_path=multichannel_spectrogram_filename, close=close_after_saving)

        for conv_id, conv_map in conv_output.items():
            conv_filename = output_dir_path / f"{sample_idx}_{conv_id}.png"
            plot_multichannel_spectrogram(
                conv_map[i], unwrap=unwrap, output_path=conv_filename, mode="row",
                figsize=(5, 10), close=close_after_saving)
        
        rnn_filename = output_dir_path / f"{sample_idx}_rnn_output.png"
        plot_spectrogram(rnn_output[i], unwrap=unwrap, output_path=rnn_filename, close=close_after_saving)
