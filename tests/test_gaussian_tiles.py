import numpy as np
from experiments.gaussian_tiles import predict_stack


def test_gaussian_tiles_preserve_constant_and_native_shape():
    x=np.zeros((2,37,43),np.float32)
    y=predict_stack(x,lambda b:np.full((len(b),1,32,32),.37,np.float32),1,32,16)
    assert y.shape==x.shape
    np.testing.assert_allclose(y,.37,atol=2e-7)


def test_gaussian_tiles_reduce_synthetic_border_error():
    x=np.zeros((1,48,48),np.float32)
    def predict(b):
        p=np.ones((len(b),1,32,32),np.float32)
        p[:,:,:4,:]=0;p[:,:,-4:,:]=0;p[:,:,:,:4]=0;p[:,:,:,-4:]=0
        return p
    from segmentation.focus.model import predict_stack as uniform
    weighted=predict_stack(x,predict,1,32,16)
    baseline=uniform(x,predict,1,32,16)
    assert weighted[:,16:32,16:32].mean()>baseline[:,16:32,16:32].mean()
