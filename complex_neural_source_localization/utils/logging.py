import torch

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
            feature_maps = feature_maps.unsqueeze(1).repeat([1, 3, 1, 1])
            feature_maps[:, 0:2, :, :] = 0

            if feature_maps.dtype == torch.complex64:
                feature_maps_mag = feature_maps.abs()
                feature_maps_phase = feature_maps.angle()
                # TODO: Phase unwrapping

                feature_maps_mag = make_grid(feature_maps_mag, normalize=True, padding=5)
                self.trainer.logger.experiment.add_image(f"{layer}.mag.epoch{n_epoch}", feature_maps_mag,)
                feature_maps_phase = make_grid(feature_maps_phase, normalize=True, padding=5)
                self.trainer.logger.experiment.add_image(f"{layer}.phase.epoch{n_epoch}", feature_maps_phase)
            else:
                feature_maps = make_grid(feature_maps, normalize=True, padding=5)
                
                self.trainer.logger.experiment.add_image(f"{layer}.epoch{n_epoch}", feature_maps)

    def _create_hook(self, layer_id):
        def fn(_, __, output):
            self.feature_maps[layer_id] = output
        return fn
