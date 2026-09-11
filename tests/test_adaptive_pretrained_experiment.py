import torch
from experiments.adaptive_pretrained import AdaptiveResNetUNet


def test_encoder_statistics_adapt_in_training_and_freeze_in_evaluation():
    torch.set_num_threads(2)
    model=AdaptiveResNetUNet().train()
    bn=model.encoder.bn1
    before=bn.running_mean.clone()
    x=torch.rand(2,1,64,64)
    y=model(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    assert not torch.equal(before,bn.running_mean)
    model.eval()
    before=bn.running_mean.clone()
    with torch.no_grad(): model(x)
    torch.testing.assert_close(before,bn.running_mean,rtol=0,atol=0)
