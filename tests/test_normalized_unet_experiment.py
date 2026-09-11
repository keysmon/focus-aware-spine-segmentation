import torch
from segmentation.model import SmallUNet
from experiments.normalized_unet import NormalizedUNet


def test_normalization_preserves_initial_convolution_weights():
    torch.manual_seed(435)
    baseline = SmallUNet()
    torch.manual_seed(435)
    candidate = NormalizedUNet()
    a = [m.weight for m in baseline.modules() if isinstance(m, torch.nn.Conv2d)]
    b = [m.weight for m in candidate.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_predictions_do_not_depend_on_other_batch_members_and_reload(tmp_path):
    torch.set_num_threads(2)
    model = NormalizedUNet().eval()
    x = torch.rand(1, 1, 32, 40)
    with torch.no_grad():
        a = model(x)
        b = model(torch.cat([x, torch.zeros_like(x)]))[:1]
    assert a.shape == x.shape and torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    path = tmp_path/'weights.pt'
    torch.save(model.state_dict(), path)
    restored = NormalizedUNet().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(a, restored(x), rtol=0, atol=0)
