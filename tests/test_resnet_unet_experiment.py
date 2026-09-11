import torch
import pytest
from experiments.resnet_unet import ResNetUNet


def test_resnet_decoder_preserves_size_and_frozen_encoder_statistics():
    torch.set_num_threads(2)
    model=ResNetUNet().train()
    assert all(not m.training for m in model.encoder.modules() if isinstance(m,torch.nn.BatchNorm2d))
    before=model.encoder.bn1.running_mean.clone()
    x=torch.rand(1,1,33,45)
    y=model(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    y.mean().backward()
    assert model.encoder.conv1.weight.grad is not None
    assert torch.isfinite(model.encoder.conv1.weight.grad).all()
    torch.testing.assert_close(before,model.encoder.bn1.running_mean,rtol=0,atol=0)


def test_resnet_rejects_unknown_external_weight_files(tmp_path):
    path=tmp_path/'wrong.pt'
    path.write_bytes(b'not approved checkpoint bytes')
    with pytest.raises(ValueError,match='hash'):
        ResNetUNet(weights=path)
