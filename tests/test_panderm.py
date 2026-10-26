"""Checkpoint compatibility must never silently leave random encoder weights."""
import pytest
import torch

from src.models.panderm import validate_encoder_state


def test_encoder_loads_all_weights_and_ignores_only_classifier():
    model = torch.nn.Linear(3, 2)
    weights = {'weight': torch.ones(2, 3), 'bias': torch.ones(2),
               'head.weight': torch.zeros(7, 2), 'head.bias': torch.zeros(7)}
    validate_encoder_state(model, weights)
    torch.testing.assert_close(model(torch.ones(1, 3)), torch.full((1, 2), 4.))


@pytest.mark.parametrize('state', [
    {'weight': torch.ones(2, 3)},
    {'weight': torch.ones(2, 3), 'bias': torch.ones(2), 'encoder.extra': torch.ones(1)},
    {'weight': torch.ones(2, 4), 'bias': torch.ones(2)},
    {'model': {'weight': torch.ones(2, 3)}}])
def test_encoder_rejects_incomplete_or_incompatible_weights(state):
    with pytest.raises(ValueError):
        validate_encoder_state(torch.nn.Linear(3, 2), state)
