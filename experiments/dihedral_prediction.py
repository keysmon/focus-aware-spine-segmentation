"""Average eight exact square symmetries, restoring output coordinates."""
import numpy as np


def dihedral_predictor(predict):
    def wrapped(batch):
        output=[]
        for k in range(4):
            for flip in (False,True):
                x=np.rot90(batch,k,axes=(-2,-1))
                if flip:x=np.flip(x,-1)
                y=predict(np.ascontiguousarray(x))
                if flip:y=np.flip(y,-1)
                output.append(np.rot90(y,-k,axes=(-2,-1)))
        return np.mean(output,axis=0,dtype=np.float32)
    return wrapped
