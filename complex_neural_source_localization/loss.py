from torch.nn import Module, CosineSimilarity


class AngularLoss(Module):
    def __init__(self, model_output_type="scalar"):
        # See https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        # for a related implementation used for NLP
        super().__init__()

        dim = 1 if model_output_type == "scalar" else 2
        self.cosine_similarity = CosineSimilarity(dim=dim)
        # dim=0 -> batch | dim=1 -> time steps | dim=2 -> azimuth

    def forward(self, model_output, targets, mean_reduce=True):
        values = 1 - self.cosine_similarity(model_output, targets["azimuth_2d_point"])
        if mean_reduce:
            values = values.mean()
        return values
