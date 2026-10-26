"""Frozen PanDerm Base using the authors' pinned encoder implementation."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from urllib.request import urlopen

import torch
from torchvision import transforms

UPSTREAM_COMMIT = 'fd7a80748ba7fc3e203fed88f909f4689d0d6f24'
SOURCE_SHA256 = '63fbb03283bb5d61ebd24e4790b66f3495996cc37f000f7003eec0a277bd1ab1'
SOURCE_URL = f'https://raw.githubusercontent.com/SiyuanYan1/PanDerm/{UPSTREAM_COMMIT}/classification/models/modeling_finetune.py'
CHECKPOINT_ID = '17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def get_source(path='artifacts/PanDerm/classification/models/modeling_finetune.py'):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
            raise ValueError('Upstream source checksum mismatch')
        path.write_bytes(payload)
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError('PanDerm encoder source differs from the reviewed pinned revision')
    return path


def validate_encoder_state(model, checkpoint):
    """Reject partially/randomly initialized encoders; discard classifier only."""
    if not isinstance(checkpoint, dict) or not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        raise ValueError('Expected a flat tensor-only PanDerm encoder checkpoint')
    state = {key: value for key, value in checkpoint.items() if not key.startswith('head.')}
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f'Encoder keys mismatch: missing={sorted(set(expected)-set(state))}, unexpected={sorted(set(state)-set(expected))}')
    if any(state[key].shape != expected[key].shape for key in state):
        raise ValueError('Encoder tensor dimensions do not match PanDerm Base')
    model.load_state_dict(state, strict=True)


def build_panderm(checkpoint_path=None, variant='base'):
    if variant not in {'base', 'large'}:
        raise ValueError('PanDerm variant must be base or large')
    if checkpoint_path is None:
        checkpoint_path = 'artifacts/panderm_base.pth' if variant == 'base' else 'artifacts/panderm_paper/panderm_large.pth'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = get_source()
    name = '_project_panderm_encoder'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    factory = (sys.modules[name].panderm_base_patch16_224 if variant == 'base'
               else sys.modules[name].panderm_large_patch16_224)
    model = factory()
    model.head = torch.nn.Identity()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if variant == 'large' and any(key.startswith('encoder.') for key in checkpoint):
        # Original release includes pretraining-only modules. The official
        # builder strips encoder. and ignores unused modules with strict=False;
        # require complete encoder coverage here instead of silent partial loads.
        encoder_state = {key.removeprefix('encoder.'): value for key, value in checkpoint.items()
                         if key.startswith('encoder.')}
        model.pretraining_only_keys = [key for key in checkpoint if not key.startswith('encoder.')]
    else:
        encoder_state = checkpoint
        model.pretraining_only_keys = []
    validate_encoder_state(model, encoder_state)
    model.requires_grad_(False).eval()
    # Exactly match get_eval_transforms in upstream models/builder.py.
    # The red-channel std really is .228 in that implementation, not .229.
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.228, .224, .225))])
    return model, transform
