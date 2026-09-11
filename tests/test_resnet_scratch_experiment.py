import torch
from experiments.resnet_scratch import ScratchResNetUNet


def test_scratch_encoder_statistics_update_only_during_training():
    torch.set_num_threads(2)
    model=ScratchResNetUNet()
    x=torch.rand(2,1,32,40)
    before=model.encoder.bn1.running_mean.clone()
    model(x)
    assert not torch.equal(before,model.encoder.bn1.running_mean)
    model.eval()
    frozen=model.encoder.bn1.running_mean.clone()
    with torch.no_grad():
        out=model(x[:1])
    assert out.shape==x[:1].shape
    torch.testing.assert_close(frozen,model.encoder.bn1.running_mean,rtol=0,atol=0)
