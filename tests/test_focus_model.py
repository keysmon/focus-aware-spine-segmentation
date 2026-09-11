import numpy as np
import pytest
import torch
from segmentation.focus.model import FocusUNet, predict_stack, load_predictor


@pytest.mark.parametrize('channels',[1,5])
def test_model_shape_and_checkpoint(tmp_path,channels):
    torch.set_num_threads(2)
    model=FocusUNet(channels).eval()
    x=torch.zeros(1,channels,32,40)
    with torch.no_grad(): expected=model(x).sigmoid().numpy()
    assert expected.shape==(1,1,32,40)
    path=tmp_path/'best.pt'
    torch.save(dict(model=model.state_dict(),channels=channels),path)
    np.testing.assert_allclose(load_predictor(path,'cpu')(x.numpy()),expected,atol=1e-6)


def test_center_channel_identity_reconstruction():
    for shape in [(3,137,151),(2,17,23)]:
        x=np.random.default_rng(435).random(shape,dtype=np.float32)
        p=predict_stack(x,lambda batch: batch[:,2:3],5)
        np.testing.assert_allclose(p,x,atol=1e-6)


def test_bad_context_and_probability_rejected():
    with pytest.raises(ValueError): FocusUNet(3)
    with pytest.raises(ValueError): predict_stack(np.zeros((1,16,16),np.float32),lambda b: b[:,2:3]+2,5)
