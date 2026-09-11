"""Matched recipes; selection sees only the supplied validation pair."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from segmentation.train import device_name, loss_value
from segmentation.focus.data import normalize_stack, Sampler
from segmentation.focus.model import FocusUNet, predictor, predict_stack
from segmentation.focus.metrics import selection_score


def train_fold(pairs,validation,config,channels,output,deadline=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(config['seed'])
    device=device_name(); model=FocusUNet(channels).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    normalized=[(normalize_stack(x)[0],y) for x,y in pairs]
    sampler=Sampler(normalized,config['seed'],config['patch_size'])
    vx,vy=validation; vx=normalize_stack(vx)[0]
    history=[]; best=(-1.,-1.); selected=None; stale=0; train_seconds=0.
    started=time.monotonic(); status='epoch_limit'; steps=0
    for epoch in range(config['epochs']):
        model.train(); losses=[]
        for _ in range(config['batches_per_epoch']):
            if train_seconds>=config['training_seconds'] or (deadline and time.monotonic()>=deadline):
                status='time_limit'; break
            tick=time.monotonic()
            batch=[sampler.sample(channels) for _ in range(config['batch_size'])]
            x=torch.from_numpy(np.asarray([a for a,_,_ in batch])).to(device)
            y=torch.from_numpy(np.asarray([b for _,b,_ in batch],np.float32)[:,None]).to(device)
            optimizer.zero_grad(); loss=loss_value(model(x),y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            steps+=1; train_seconds+=time.monotonic()-tick
        if not losses: break
        probabilities=predict_stack(vx,predictor(model,device),channels,config['patch_size'],config['stride'])
        options=[(t,selection_score(vy,probabilities>=t)) for t in config['thresholds']]
        threshold,score=max(options,key=lambda item:item[1])
        record=dict(epoch=epoch+1,steps=steps,loss=float(np.mean(losses)),validation_dice=score[0],
                    validation_empty_fp=-score[1],threshold=threshold,training_seconds=train_seconds,
                    schedule_hash=sampler.schedule_hash(),sampling_counts=sampler.counts.copy())
        history.append(record)
        print(json.dumps(dict(channels=channels,**record)),flush=True)
        if score>best:
            best=score; selected=record; stale=0
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),channels=channels,
                            config=config,selected=record,normalization='stack-percentile-1-99',
                            sampler='equal-foreground-hardnegative-uniform',device=device),output/'best.tmp')
            (output/'best.tmp').replace(output/'best.pt')
        else: stale+=1
        (output/'history.json').write_text(json.dumps(history,indent=2))
        if status=='time_limit': break
        if stale>=config['patience']:
            status='early_stopping'; break
    result=dict(channels=channels,status=status,selected=selected,history=history,device=device,
                parameter_count=sum(p.numel() for p in model.parameters()),training_seconds=train_seconds,
                elapsed_seconds=time.monotonic()-started,sampling_counts=sampler.counts)
    (output/'selection.json').write_text(json.dumps(result,indent=2))
    if selected is None: raise RuntimeError('No validated checkpoint produced')
    return result


def smoke(output):
    """Identical neighbors, distinct centers (including empty centers)."""
    torch.set_num_threads(2); torch.manual_seed(435)
    device=device_name(); model=FocusUNet(5).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    x=torch.zeros(4,5,64,64,device=device)
    x[:,:,:,24:40]=.7
    x[:,2]=0
    x[0,2,8:28,8:28]=1
    x[1,2,36:56,36:56]=1
    target=x[:,2:3].clone()
    start=time.monotonic(); first=None; dice=0.; empty_fp=1.
    for step in range(200):
        model.train(); optimizer.zero_grad(); loss=loss_value(model(x),target)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
        loss.backward(); optimizer.step()
        first=float(loss.detach()) if first is None else first
        with torch.no_grad():
            p=model(x).sigmoid()>=.5
            dice=float(2*(p*target).sum()/(p.sum()+target.sum()))
            empty_fp=float(p[2:].float().mean())
        if step>=19 and dice>=.9 and empty_fp<=.01 and float(loss.detach())<first: break
        if time.monotonic()-start>=120: break
    with torch.no_grad():
        swapped=x.clone(); swapped[:,2]=0
        zeroed_foreground=float((model(swapped).sigmoid()>=.5).float().mean())
    result=dict(steps=step+1,dice=dice,empty_fp=empty_fp,zeroed_center_foreground=zeroed_foreground,
                first_loss=first,last_loss=float(loss.detach()),seconds=time.monotonic()-start,device=device)
    Path(output).write_text(json.dumps(result,indent=2))
    if dice<.9 or empty_fp>.01 or zeroed_foreground>.01:
        raise RuntimeError('Center-dependent smoke failed; inspect smoke.json')
    return result
