"""Paired training transforms and overlap-averaged full-resolution inference."""
import numpy as np


def augment_pair(image, mask, rng):
    k = int(rng.integers(4))
    image, mask = np.rot90(image, k), np.rot90(mask, k)
    for axis in (0, 1):
        if rng.random() < 0.5:
            image, mask = np.flip(image, axis), np.flip(mask, axis)
    image = np.clip(image * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask, dtype=bool)


def sample_patch(images, masks, rng, size=128):
    index = int(rng.integers(len(images)))
    x, y = images[index], masks[index]
    if rng.random() < 0.5:
        indices = np.flatnonzero(masks.reshape(len(masks), -1).any(axis=1))
        if len(indices):
            index = int(rng.choice(indices)); x, y = images[index], masks[index]
            coords = np.argwhere(y)
            cy, cx = coords[int(rng.integers(len(coords)))]
        else:
            cy, cx = int(rng.integers(x.shape[0])), int(rng.integers(x.shape[1]))
    else:
        cy, cx = int(rng.integers(x.shape[0])), int(rng.integers(x.shape[1]))
    padding = ((0, max(0, size - x.shape[0])), (0, max(0, size - x.shape[1])))
    x = np.pad(x, padding, mode='edge'); y = np.pad(y, padding)
    top = int(np.clip(cy - size // 2, 0, x.shape[0] - size))
    left = int(np.clip(cx - size // 2, 0, x.shape[1] - size))
    return augment_pair(x[top:top+size, left:left+size], y[top:top+size, left:left+size], rng)


def predict_tiled(images, predict_batch, size=128, stride=64):
    if not 0 < stride <= size:
        raise ValueError('stride must be positive and no greater than patch size')
    results = []
    for frame in images:
        h, w = frame.shape
        padded = np.pad(frame, ((0, max(0, size-h)), (0, max(0, size-w))), mode='edge')
        def starts(length):
            return sorted(set([*range(0, length-size+1, stride), length-size]))
        coords = [(y, x) for y in starts(padded.shape[0]) for x in starts(padded.shape[1])]
        total = np.zeros_like(padded, dtype=np.float32)
        count = np.zeros_like(total)
        for start in range(0, len(coords), 4):
            group = coords[start:start+4]
            batch = np.asarray([padded[y:y+size, x:x+size] for y, x in group], np.float32)[:, None]
            probabilities = predict_batch(batch)
            if probabilities.shape != batch.shape or not np.isfinite(probabilities).all():
                raise ValueError('Predictor must return finite probabilities with input shape')
            for (y, x), probability in zip(group, probabilities[:, 0]):
                total[y:y+size, x:x+size] += probability
                count[y:y+size, x:x+size] += 1
        results.append((total / count)[:h, :w])
    return np.asarray(results, np.float32)
