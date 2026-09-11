"""Bounded training with validation-only selection and atomic checkpoints."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from segmentation.classical import normalize
from segmentation.metrics import evaluate
from segmentation.model import SmallUNet
from segmentation.patches import sample_patch, predict_tiled


def device_name():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def loss_value(logits, target):
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target) + dice


def predictor(model, device):
    def predict(batch):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(batch).to(device)).sigmoid().cpu().numpy()
    return predict


def load_predictor(checkpoint: Path, device: str):
    model = SmallUNet().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    return predictor(model, device)


def smoke(output):
    """Overfit a simple aligned geometric target before using real data."""
    torch.set_num_threads(2); torch.manual_seed(435)
    device = device_name()
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    target = torch.zeros(2, 1, 128, 128, device=device)
    target[:, :, 32:80, 40:88] = 1
    x = target.clone()
    started = time.monotonic(); first_loss = None; benchmark = None
    for step in range(200):
        optimizer.zero_grad(); logits = model(x); loss = loss_value(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke loss')
        loss.backward(); optimizer.step()
        if first_loss is None: first_loss = float(loss.detach())
        if step == 19: benchmark = time.monotonic() - started
        with torch.no_grad():
            pred = model(x).sigmoid() >= 0.5
            dice = float(2 * (pred * target).sum() / (pred.sum() + target.sum()))
        if step >= 19 and dice >= 0.9 and float(loss.detach()) < first_loss: break
        if time.monotonic() - started >= 120: break
    result = dict(device=device, steps=step+1, dice=dice, first_loss=first_loss,
                  last_loss=float(loss.detach()), benchmark_20_steps_seconds=benchmark,
                  seconds=time.monotonic()-started)
    Path(output).write_text(json.dumps(result, indent=2))
    if dice < 0.9 or result['last_loss'] >= first_loss:
        raise RuntimeError('Synthetic overfit check failed; inspect smoke.json')
    return result


def train_fold(train_pairs, validation_pair, config, output: Path, deadline=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    device = device_name(); model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    pairs = [(normalize(x), y) for x, y in train_pairs]
    vx, vy = validation_pair; vx = normalize(vx)
    best = -1.; stale = 0; training_seconds = 0.; history = []; selected = None
    status = 'epoch_limit'; started = time.monotonic()
    for epoch in range(config['epochs']):
        losses = []; model.train()
        for _ in range(config['batches_per_epoch']):
            if training_seconds >= config['training_seconds'] or (deadline and time.monotonic() >= deadline):
                status = 'time_limit'; break
            tick = time.monotonic()
            batch = [sample_patch(*pairs[int(rng.integers(len(pairs)))], rng, size=config['patch_size']) for _ in range(config['batch_size'])]
            x = torch.from_numpy(np.asarray([a for a, _ in batch])[:, None]).to(device)
            y = torch.from_numpy(np.asarray([b for _, b in batch], np.float32)[:, None]).to(device)
            optimizer.zero_grad(); loss = loss_value(model(x), y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            training_seconds += time.monotonic()-tick
        if not losses: break
        probabilities = predict_tiled(vx, predictor(model, device), config['patch_size'], config['stride'])
        scores = [(t, evaluate(vy, probabilities >= t)['dice']) for t in config['thresholds']]
        threshold, score = max(scores, key=lambda item: item[1])
        record = dict(epoch=epoch+1, loss=float(np.mean(losses)), validation_dice=score,
                      threshold=threshold, training_seconds=training_seconds)
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best = score; stale = 0; selected = record
            temporary = output / 'best.tmp'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                            epoch=epoch+1, threshold=threshold, device=device), temporary)
            temporary.replace(output / 'best.pt')
        else: stale += 1
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        if status == 'time_limit': break
        if stale >= config['patience']:
            status = 'early_stopping'; break
    result = dict(status=status, selected=selected, device=device, training_seconds=training_seconds,
                  elapsed_seconds=time.monotonic()-started, history=history)
    (output / 'selection.json').write_text(json.dumps(result, indent=2))
    if selected is None: raise RuntimeError('No validated checkpoint produced')
    return result
