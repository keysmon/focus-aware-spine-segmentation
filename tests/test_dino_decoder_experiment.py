import torch
from experiments.dino_decoder import prepare_input,DenseDecoder


def test_preprocessing_retains_rectangular_geometry_and_constant_borders():
    x=torch.full((1,1,19,23),.5)
    p=prepare_input(x)
    assert p.shape==(1,3,42,56)
    expected=(torch.tensor(.5)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
    torch.testing.assert_close(p[0,:,0,0],expected)
    torch.testing.assert_close(p[0,:,-1,-1],expected)


def test_decoder_uses_features_and_raw_pixels_through_image_edges():
    torch.set_num_threads(2);torch.manual_seed(435)
    model=DenseDecoder()
    features=torch.randn(1,384,3,4,requires_grad=True)
    raw=torch.rand(1,1,19,23,requires_grad=True)
    y=model(features,raw)
    assert y.shape==raw.shape and torch.isfinite(y).all()
    y[:,:,-1,-1].sum().backward()
    assert features.grad[:,:,-1,-1].abs().sum()>0
    assert raw.grad[:,:,-1,-1].abs().sum()>0
