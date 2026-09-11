import numpy as np
import torch
from experiments.union_auxiliary import UnionSampler,UnionUNet,loss_value
from segmentation.focus.data import Sampler
from experiments.tversky import loss_value as primary_loss


def test_union_target_shares_transform_and_preserves_primary_mask():
    x=np.random.default_rng(2).random((3,32,32),dtype=np.float32)
    y=np.zeros(x.shape,bool);y[0,4:12,7:15]=True;y[1,15:23,19:27]=True
    a=Sampler([(x,y)],435,32);b=UnionSampler([(x,y)],435,32)
    for _ in range(12):
        image,mask,meta=a.sample(1);image2,targets,meta2=b.sample(1)
        assert np.array_equal(image,image2) and meta==meta2
        assert np.array_equal(mask,targets[0])
        assert np.all(~targets[0]|targets[1])
        assert targets[1].sum()==128
    assert y.sum()==128


def test_auxiliary_head_is_excluded_from_inference():
    torch.set_num_threads(2);model=UnionUNet(1).eval();x=torch.rand(2,1,33,41)
    with torch.no_grad():
        both=model.forward_heads(x);focus=model(x)
    assert both.shape==(2,2,33,41) and focus.shape==(2,1,33,41)
    torch.testing.assert_close(focus,both[:,:1])


def test_auxiliary_loss_trains_both_heads_without_changing_primary_definition():
    logits=torch.randn(2,2,8,8,requires_grad=True);target=torch.zeros_like(logits)
    target[:,0,2:4,2:4]=1;target[:,1,1:6,1:6]=1
    torch.testing.assert_close(loss_value(logits,target,weight=0),primary_loss(logits[:,:1],target[:,:1]))
    loss=loss_value(logits,target);loss.backward()
    assert torch.isfinite(loss) and logits.grad[:,0].abs().sum()>0 and logits.grad[:,1].abs().sum()>0
