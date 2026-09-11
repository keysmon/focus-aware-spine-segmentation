"""Wider bottleneck context with unchanged learned parameter tensors."""
from segmentation.focus.model import FocusUNet


class DilatedCenterUNet(FocusUNet):
    def __init__(self,channels):
        super().__init__(channels)
        for index,rate in ((0,2),(2,4)):
            self.center[index].dilation=(rate,rate)
            self.center[index].padding=(rate,rate)
