from inspect import unwrap
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from pathlib import Path
from torchvision.utils import make_grid
from tqdm import tqdm

from complex_neural_source_localization.utils.model_utilities import (
    get_all_layers
)
from complex_neural_source_localization.utils.conv_block import ConvBlock


def plot_multichannel_spectrogram(multichannel_spectrogram, unwrap=True, unwrap_mode="freq", mode="column",
                                  axs=None, figsize=(10, 5), output_path=None, close=True, db=True,
                                  colorbar=True):
    
    num_channels, num_freq_bins, num_time_steps = multichannel_spectrogram.shape
    
    if axs is None:
        if mode == "column":
            n_rows, n_cols = (2, num_channels)
            share_y = "col"
        elif mode == "row":
            n_rows, n_cols = (num_channels, 2)
            share_y = "row"

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, sharey=share_y)

    # If torch.Tensor, move to cpu and convert to numpy
    if isinstance(multichannel_spectrogram, torch.Tensor):
        multichannel_spectrogram = multichannel_spectrogram.cpu().detach().numpy()
    
    # Plot spectrograms for all channels
    for n_channel in range(num_channels):
        if mode == "row":
            channel_axs = [axs[n_channel, 0], axs[n_channel, 1]]
        elif mode == "column":
            channel_axs = [axs[0, n_channel], axs[1, n_channel]]

        else:
            raise ValueError("Allowed modes are 'row' and 'column'")

        (mag_mesh, phase_mesh), _ = plot_spectrogram(multichannel_spectrogram[n_channel],
                                                axs=channel_axs, unwrap=unwrap, unwrap_mode=unwrap_mode,
                                                db=db, colorbar=False)

    if colorbar:
        if mode == "column":
            location = "right"
            mag_axs, phase_axs = axs[0, :], axs[1, :]
        elif mode == "row":
            location = "top"
            mag_axs, phase_axs = axs[:, 0], axs[:, 1]
        
        plt.colorbar(mag_mesh, ax=mag_axs, format="%+2.f",
                        location=location)
        plt.colorbar(phase_mesh, ax=phase_axs,
                        location=location)

    if output_path is not None:
        plt.savefig(output_path)
        if close:
            plt.close()

    return axs


def plot_spectrogram(spectrogram, unwrap=True, unwrap_mode="freq", db=True,
                     mode="column", figsize=(10, 5), axs=None, output_path=None, close=True,
                     colorbar=True):
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
    spectrogram_mag = np.abs(spectrogram)
    if db:
        spectrogram_mag = librosa.amplitude_to_db(spectrogram_mag, ref=np.max)
    spectrogram_phase = np.angle(spectrogram)
    if unwrap:
        axis = 0 if unwrap_mode == "freq" else 1
        spectrogram_phase = np.unwrap(spectrogram_phase, axis=axis)
    
    # img_mag = librosa.display.specshow(spectrogram_mag, ax=axs[0])
    # img_phase = librosa.display.specshow(spectrogram_phase, ax=axs[1])

    # https://matplotlib.org/stable/tutorials/colors/colormaps.html for beautiful colormaps
    mag_mesh = axs[0].pcolormesh(spectrogram_mag, cmap="RdBu_r")
    phase_mesh = axs[1].pcolormesh(spectrogram_phase, cmap="RdBu_r")

    axs[0].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])

    if colorbar:
        plt.colorbar(mag_mesh, ax=axs[0], format="%+2.f dB")
        plt.colorbar(phase_mesh, ax=axs[1], format="%+4.f rad")

    if output_path is not None:
        plt.savefig(output_path)
        if close:
            plt.close()
    
    return (mag_mesh, phase_mesh), axs


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
    rnn_output = feature_maps["rnn"].transpose(1, 2)

    for i in tqdm(range(batch_size)):
        sample_idx = batch_size + batch_start_idx
        multichannel_spectrogram_filename = output_dir_path / f"{sample_idx}_multichannel_stft.png"

        plot_multichannel_spectrogram(
            stft_output[i], unwrap=unwrap,
            output_path=multichannel_spectrogram_filename, close=close_after_saving)

        for conv_id, conv_map in conv_output.items():
            if conv_map.is_complex():
                conv_filename = output_dir_path / f"{sample_idx}_{conv_id}.png"
                plot_multichannel_spectrogram(
                    conv_map[i], unwrap=unwrap, output_path=conv_filename, mode="row",
                    figsize=(5, 10), close=close_after_saving)
            else:
                print("real")
        
        rnn_filename = output_dir_path / f"{sample_idx}_rnn_output.png"
        plot_spectrogram(rnn_output[i], unwrap=unwrap, output_path=rnn_filename, close=close_after_saving)


def plot_real_feature_maps(feature_maps, mode="column", axs=None, figsize=(10, 5),
                          output_path=None, close=True):
    
    num_channels, num_freq_bins, num_time_steps = feature_maps.shape
    
    if axs is None:
        if mode == "column":
            n_rows, n_cols = (1, num_channels)
        elif mode == "row":
            n_rows, n_cols = (num_channels, 1)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    # If torch.Tensor, move to cpu and convert to numpy
    if isinstance(feature_maps, torch.Tensor):
        multichannel_spectrogram = feature_maps.cpu().detach().numpy()
    
    # Plot spectrograms for all channels
    for n_channel in range(num_channels):
        channel_ax = axs[n_channel]

        plot_real_feature_map(multichannel_spectrogram[n_channel], ax=channel_ax)

    if output_path is not None:
        plt.savefig(output_path)
        if close:
            plt.close()

    return axs


def plot_real_feature_map(feature_map, mode="column", figsize=(10, 5), ax=None, output_path=None, close=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # If torch.Tensor, move to cpu and convert to numpy
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.cpu().detach().numpy()

    
    librosa.display.specshow(feature_map, ax=ax)

    if output_path is not None:
        plt.savefig(output_path)
        if close:
            plt.close()
    
    return ax


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
