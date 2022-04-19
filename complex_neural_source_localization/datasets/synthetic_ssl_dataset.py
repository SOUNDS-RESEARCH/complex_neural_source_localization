""" The idea of having two dataset loaders (SydraDataset and Dataset)
is because in theory the random generator implemented in Sydra could be reused for different
applications using two microphones, while this generator is specific for this application.
"""


from complex_neural_source_localization.datasets.sydra_dataset import SydraDataset


class SyntheticSSLDataset(SydraDataset):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)

    def __getitem__(self, index):

        (x, y) = super().__getitem__(index)
        
        y = {
            "mic_coordinates": y["mic_coordinates"][:, :2],
            "source_coordinates": y["source_coordinates"][:2]
        }

        return (x, y)
