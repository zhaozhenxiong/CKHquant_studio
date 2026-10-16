"""Strict pinned materialization and factual execution evidence, without model assets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from my_strategy.services import kronos_prediction as prediction


@pytest.fixture
def tiny_assets(tmp_path, monkeypatch):
    vendor = tmp_path / 'resources' / 'vendor' / 'kronos'
    vendor.mkdir(parents=True)
    (vendor / '__init__.py').write_text('''
import torch
from torch import nn
class Kronos(nn.Module):
    def __init__(self, extra_meta_buffer=False):
        super().__init__()
        self.projection = nn.Linear(4, 3)
        self.initialization_device = str(self.projection.weight.device)
        self.register_buffer('basis', torch.arange(3))
        if extra_meta_buffer:
            self.register_buffer('unprovided', torch.ones(2), persistent=False)
class KronosTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.register_buffer('group_codebook', torch.arange(12).reshape(3,4), persistent=False)
def auto_regressive_inference(*args, **kwargs):
    raise AssertionError('Load test must never run inference')
''')
    assets = tmp_path / 'assets'
    (assets / 'tokenizer').mkdir(parents=True)
    (assets / 'config.json').write_text('{}')
    (assets / 'tokenizer/config.json').write_text('{}')
    main = {'projection.weight': torch.arange(12, dtype=torch.float32).reshape(3, 4),
            'projection.bias': torch.tensor([.2, .4, .8]), 'basis': torch.tensor([7, 8, 9])}
    tokenizer = {'projection.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3),
                 'projection.bias': torch.tensor([.1, .9])}
    save_file(main, str(assets / 'model.safetensors'))
    save_file(tokenizer, str(assets / 'tokenizer/model.safetensors'))
    monkeypatch.setattr(prediction, 'runtime_resource_root', lambda: tmp_path / 'resources')
    monkeypatch.setattr(prediction, '_SINGLETON', {})
    # Avoid touching any accelerator allocator from a CPU-only fixture.
    monkeypatch.setattr(prediction, 'release_prediction_resources', prediction._SINGLETON.clear)
    return assets, main, tokenizer


def test_meta_model_strictly_assigns_every_weight_and_keeps_tokenizer_codebook(tiny_assets):
    assets, expected, token_expected = tiny_assets
    rng = torch.get_rng_state().clone()
    model, tokenizer, inference, reused = prediction._load_model(assets, 'cpu', {'assets': [], 'source': {}})
    assert not reused and model.initialization_device == 'meta'
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
    for key, value in tokenizer.state_dict().items():
        assert torch.equal(value, token_expected[key])
    assert torch.equal(tokenizer.group_codebook, torch.arange(12).reshape(3, 4))
    assert not any(value.is_meta for value in (*model.parameters(), *model.buffers(),
                                               *tokenizer.parameters(), *tokenizer.buffers()))
    assert all(not value.requires_grad for value in (*model.parameters(), *tokenizer.parameters()))
    assert not model.training and not tokenizer.training
    second = prediction._load_model(assets, 'cpu', {'assets': [], 'source': {}})
    assert second == (model, tokenizer, inference, True)


def test_missing_pinned_parameter_fails_strict_loading(tiny_assets):
    assets, expected, _ = tiny_assets
    del expected['projection.bias']
    save_file(expected, str(assets / 'model.safetensors'))
    with pytest.raises(RuntimeError, match='Missing key'):
        prediction._load_model(assets, 'cpu', {'assets': [], 'source': {}})
    assert not prediction._SINGLETON


def test_extra_unmaterialized_meta_buffer_cannot_reach_inference(tiny_assets):
    assets, _, _ = tiny_assets
    (assets / 'config.json').write_text(json.dumps({'extra_meta_buffer': True}))
    with pytest.raises(ValueError, match='materialize every'):
        prediction._load_model(assets, 'cpu', {'assets': [], 'source': {}})
    assert not prediction._SINGLETON


def test_execution_proof_checks_real_parameter_and_input_devices():
    model = torch.nn.Linear(2, 2)
    tokenizer = torch.nn.Linear(2, 2)
    inputs = (torch.ones(1, 3, 2), torch.ones(1, 3, 5), torch.ones(1, 5, 5))
    proof = prediction._execution_devices(model, tokenizer, inputs, 'cpu')
    assert proof == {'model_parameter_devices': ['cpu'], 'tokenizer_parameter_devices': ['cpu'],
                     'input_tensor_devices': ['cpu'], 'tensor_devices_verified': True}
    with pytest.raises(RuntimeError, match='differ'):
        prediction._execution_devices(model, tokenizer, inputs, 'mps')
    empty = prediction._execution_devices(None, None, inputs, 'cpu')
    assert not empty['tensor_devices_verified']


def test_seeded_cpu_state_restored_on_failure():
    before = torch.get_rng_state().clone()
    with pytest.raises(InterruptedError):
        with prediction._seeded(42, 'cpu'):
            torch.rand(10)
            raise InterruptedError('isolated cancellation')
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('requested', ['mps', 'cuda', 'cuda:0'])
def test_unavailable_explicit_accelerator_never_falls_back(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='unavailable'):
        prediction._device(requested)


def test_bare_cuda_resolves_actual_ordinal_without_gpu_access(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    assert prediction._device('cuda') == 'cuda:2'
    assert prediction._device('cuda:1') == 'cuda:1'
