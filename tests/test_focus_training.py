import numpy as np
import torch
from segmentation.focus.train import train_fold
from segmentation.focus.model import load_predictor


def test_tiny_matched_fits(tmp_path):
    x=np.zeros((3,32,32),np.uint8); x[:,8:20,8:20]=255
    y=x!=0
    config=dict(seed=435,epochs=1,batches_per_epoch=2,batch_size=1,training_seconds=10,
                patience=5,learning_rate=.001,patch_size=32,stride=16,thresholds=[.3,.5,.7])
    results=[]
    for channels in (1,5):
        result=train_fold([(x,y)],(x,y),config,channels,tmp_path/str(channels))
        results.append(result)
        assert result['selected']['epoch']==1
        predict=load_predictor(tmp_path/str(channels)/'best.pt','cpu')
        assert predict(np.zeros((1,channels,32,32),np.float32)).shape==(1,1,32,32)
    assert results[0]['history'][0]['schedule_hash']==results[1]['history'][0]['schedule_hash']
