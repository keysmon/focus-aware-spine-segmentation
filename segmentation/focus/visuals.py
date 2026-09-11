"""Label-selected, method-independent focal-sweep displays."""
from pathlib import Path
import numpy as np
from segmentation.report import plt


def location(reference):
    z=int(reference.sum(axis=(1,2)).argmax())
    points=np.argwhere(reference[z])
    cy,cx=points.mean(axis=0).astype(int).tolist() if len(points) else [reference.shape[1]//2,reference.shape[2]//2]
    return dict(z=z,cy=cy,cx=cx,size=80)


def axial_panel(images,reference,predictions,where,path):
    indices=list(range(max(0,where['z']-2),min(len(images),where['z']+3)))
    size=where['size']; h,w=images.shape[1:]
    top=int(np.clip(where['cy']-size//2,0,max(0,h-size)))
    left=int(np.clip(where['cx']-size//2,0,max(0,w-size)))
    arrays={'Image':images,'Reference':reference,**predictions}
    fig,axes=plt.subplots(len(arrays),len(indices),figsize=(3*len(indices),2.6*len(arrays)),squeeze=False)
    lo,hi=np.percentile(images,(1,99))
    for row,(name,array) in enumerate(arrays.items()):
        for col,z in enumerate(indices):
            crop=array[z,top:top+size,left:left+size]
            axes[row,col].imshow(crop,cmap='gray',vmin=lo if name=='Image' else 0,vmax=hi if name=='Image' else 1)
            axes[row,col].set_title(f'{name}: slice {z}',fontsize=10)
            axes[row,col].axis('off')
    fig.suptitle('Same XY region through consecutive focal planes')
    fig.tight_layout();fig.savefig(path,dpi=100);plt.close(fig)


def overview(stacks,path):
    names=['Image','Reference','R0','R1','R2']
    fig,axes=plt.subplots(len(stacks),5,figsize=(13,3.5*len(stacks)),squeeze=False)
    for row,(sid,(x,y,predictions)) in enumerate(stacks.items()):
        z=location(y)['z']
        arrays=[x[z],y[z],predictions['R0'][z],predictions['R1'][z],predictions['R2'][z]]
        for col,(name,array) in enumerate(zip(names,arrays)):
            axes[row,col].imshow(array,cmap='gray')
            axes[row,col].set_title(f'{sid}, slice {z}: {name}',fontsize=10)
            axes[row,col].axis('off')
    fig.suptitle('Focus-aware development comparison: same preselected slices')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
