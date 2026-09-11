"""Preserved baseline and four fixed cumulative ablations."""
import cv2
import numpy as np
from scipy.ndimage import label
from project import generate_segmentation_map
from segmentation.metrics import evaluate

CANDIDATES = [dict(name='C0', normalize=False, adaptive=False, cleanup=False),
    dict(name='C1', normalize=True, adaptive=False, cleanup=False),
    dict(name='C2', normalize=True, adaptive=True, cleanup=False),
    dict(name='C3', normalize=True, adaptive=True, cleanup=True)]


def baseline(images):
    return generate_segmentation_map(images)


def normalize(images):
    result = []
    for frame in images:
        lo, hi = np.percentile(frame, (1, 99))
        result.append(np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros(frame.shape, np.float32))
    return np.asarray(result, np.float32)


def classical(images, config):
    if config == CANDIDATES[0]:
        return baseline(images)
    x = (normalize(images) * 255).astype(np.uint8) if config['normalize'] else images
    if min(x.shape[1:]) < 31:
        raise ValueError('Adaptive candidate requires spatial dimensions at least 31')
    if not config['adaptive']:
        result = baseline(x)
        result[np.ptp(x, axis=(1, 2)) == 0] = 0
        return result
    # Reproduce the baseline focus projection and object restriction.
    gradients = np.asarray([np.abs(cv2.Laplacian(a, cv2.CV_64F, ksize=1)) for a in x])
    focus = np.take_along_axis(x, gradients.argmax(axis=0)[None], axis=0)[0]
    _, inv = cv2.threshold(cv2.medianBlur(focus, 3), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(focus, inv)
    _, thresh = cv2.threshold(masked, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    highlighted = (thresh != inv).astype(np.uint8) * 255
    density = cv2.GaussianBlur(highlighted, (41, 41), 150)
    density = cv2.bitwise_and(density, density, mask=inv)
    _, object_mask = cv2.threshold(density, 3, 255, cv2.THRESH_BINARY)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    result = []
    for frame, grad in zip(x, gradients):
        if np.ptp(frame) == 0:
            result.append(np.zeros_like(frame)); continue
        mask = ((grad.astype(np.uint8) >= 20) & (object_mask != 0)).astype(np.uint8) * 255
        density = cv2.GaussianBlur(mask, (19, 19), 15)
        density = cv2.bitwise_and(density, density, mask=object_mask)
        pred = cv2.adaptiveThreshold(density, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        pred[object_mask == 0] = 0
        if config['cleanup']:
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            labels, _ = label(pred != 0)
            counts = np.bincount(labels.ravel()); counts[0] = 0
            pred = (counts[labels] >= 16).astype(np.uint8) * 255
        result.append(pred)
    return np.asarray(result, np.uint8)


def select_classical(images, reference, candidates):
    scores = [dict(config=c, metrics=evaluate(reference, classical(images, c) != 0)) for c in candidates]
    best = max(scores, key=lambda item: item['metrics']['dice'])
    return dict(selected=best['config'], validation=scores)
