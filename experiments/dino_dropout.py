"""Channel dropout regularizes the fixed pretrained feature representation."""
from torch import nn
from experiments.dino_decoder import DenseDecoder


class DropoutDecoder(DenseDecoder):
    def __init__(self):
        super().__init__()
        self.feature_dropout=nn.Dropout2d(.5)

    def forward(self,features,image):
        return super().forward(self.feature_dropout(features),image)
