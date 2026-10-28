from collections import OrderedDict

import pytest
import torch.nn as nn

from precision_planner import _sample


@pytest.fixture
def sample_plan():
    return _sample.make_sample_plan(
        _sample.DISTILBERT_SST2,
        _sample.DISTILBERT_LINEAR_LAYERS,
        _sample.DISTILBERT_OTHER_PARAMS,
        0.01,
        "nearest",
    )


@pytest.fixture
def tiny_model():
    return nn.Sequential(
        OrderedDict(
            fc1=nn.Linear(4, 8),
            act=nn.ReLU(),
            fc2=nn.Linear(8, 8),
            norm=nn.LayerNorm(8),
            fc3=nn.Linear(8, 2),
        )
    )
