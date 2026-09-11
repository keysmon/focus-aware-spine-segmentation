"""Whole-stack membership, frozen before model selection."""
from types import MappingProxyType

FOLDS = tuple(MappingProxyType(dict(test=t, validation=v, train=train)) for t, v, train in [
    ('2', '3', ('6', '7')), ('3', '6', ('2', '7')),
    ('6', '7', ('2', '3')), ('7', '2', ('3', '6')),
])
