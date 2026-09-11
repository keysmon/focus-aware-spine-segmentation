import torch
from experiments.dino_multilayer import MultiLayerDecoder


def test_all_feature_layers_contribute_to_native_output():
    torch.set_num_threads(2);torch.manual_seed(435)
    model=MultiLayerDecoder().eval()
    features=torch.randn(1,1536,3,4,requires_grad=True);image=torch.rand(1,1,19,23)
    out=model(features,image)
    assert out.shape==image.shape and torch.isfinite(out).all()
    out.sum().backward()
    for group in features.grad.split(384,dim=1):assert group.abs().sum()>0
