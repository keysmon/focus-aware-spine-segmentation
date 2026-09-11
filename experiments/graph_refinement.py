"""Binary 2D Potts graph cut with image-guided edges and model unary costs."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order


def refine_plane(image,probability,threshold=.3,strength=2.):
    x,p=np.asarray(image,np.float64),np.asarray(probability,np.float64)
    if x.shape!=p.shape or x.ndim!=2 or not x.size:
        raise ValueError('Expected matching nonempty HW arrays')
    if not np.isfinite(x).all() or not np.isfinite(p).all() or min(x.min(),p.min())<0 or max(x.max(),p.max())>1:
        raise ValueError('Expected finite image and probabilities in [0,1]')
    if not np.isfinite(strength) or strength<0 or not 0<threshold<1:
        raise ValueError('Invalid threshold or pairwise strength')
    if strength==0:return p>=threshold
    ids=np.arange(x.size).reshape(x.shape);n=x.size;source=n;sink=n+1
    a=np.concatenate([ids[:,:-1].ravel(),ids[:-1,:].ravel()])
    b=np.concatenate([ids[:,1:].ravel(),ids[1:,:].ravel()])
    differences=(x.ravel()[a]-x.ravel()[b])**2
    scale=10000
    denominator=2*differences.mean()+1e-12 if len(differences) else 1.
    weights=np.rint(scale*strength*np.exp(-differences/denominator)).astype(np.int64)
    clipped=np.clip(p.ravel(),1e-6,1-1e-6)
    score=np.log(clipped/(1-clipped))-np.log(threshold/(1-threshold))
    background=np.rint(scale*np.maximum(score,0)).astype(np.int64)
    foreground=np.rint(scale*np.maximum(-score,0)).astype(np.int64)
    pixels=ids.ravel()
    rows=np.concatenate([np.full(n,source),pixels,a,b])
    cols=np.concatenate([pixels,np.full(n,sink),b,a])
    values=np.concatenate([background,foreground,weights,weights])
    graph=csr_matrix((values,(rows,cols)),shape=(n+2,n+2));graph.eliminate_zeros()
    flow=maximum_flow(graph,source,sink)
    residual=graph-flow.flow
    residual.data[residual.data<=0]=0;residual.eliminate_zeros()
    reached=breadth_first_order(residual,source,directed=True,return_predecessors=False)
    result=np.zeros(n+2,bool);result[reached]=True
    return result[:n].reshape(x.shape)
