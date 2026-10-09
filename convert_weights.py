"""
Convert a PaddleDetection PP-YOLOE .pdparams checkpoint to a PyTorch state dict
loadable by models.yoloe.YOLOE.

Legacy .pdparams files pickle tensors as plain numpy arrays, so PaddlePaddle is
not required; the paddle module paths in the pickle are stubbed out.

Usage:
    python convert_weights.py --input weights/ppyoloe_plus_crn_s_80e_coco.pdparams \
        --output weights/ppyoloe_plus_crn_s_80e_coco.pt
"""
import argparse
import pickle
import re
import sys
import types

import numpy as np
import torch


class _FakeTensor:
    def __init__(self, *args, **kwargs):
        pass


def _stub_paddle():
    mods = {}
    for name in ['paddle', 'paddle.base', 'paddle.base.core',
                 'paddle.base.core.eager', 'paddle.base.libpaddle']:
        mods[name] = types.ModuleType(name)
        sys.modules[name] = mods[name]
    mods['paddle'].Tensor = _FakeTensor
    mods['paddle'].base = mods['paddle.base']
    mods['paddle.base'].core = mods['paddle.base.core']
    mods['paddle.base'].libpaddle = mods['paddle.base.libpaddle']
    mods['paddle.base.core'].eager = mods['paddle.base.core.eager']
    mods['paddle.base.core'].libpaddle = mods['paddle.base.libpaddle']
    mods['paddle.base.core.eager'].EagerParamBase = _FakeTensor
    mods['paddle.base.libpaddle'].Tensor = _FakeTensor


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _FakeTensor


def load_pdparams(path):
    _stub_paddle()
    with open(path, 'rb') as f:
        data = _StubUnpickler(f).load()
    if isinstance(data, dict) and 'state_dict' in data:
        data = data['state_dict']
    return {k: v for k, v in data.items() if isinstance(v, np.ndarray)}


STEM_RE = re.compile(r'^backbone\.stem\.conv(\d)\.')
FPN_RE = re.compile(r'^neck\.fpn_stages\.(\d+)\.0\.')
PAN_RE = re.compile(r'^neck\.pan_stages\.(\d+)\.0\.')
CONVS_RE = re.compile(r'\.convs\.(\d+)\.')


def map_key(key):
    key = key.replace('yolo_head.', 'head.')
    key = STEM_RE.sub(lambda m: 'backbone.stem.%d.' % (int(m.group(1)) - 1), key)
    key = FPN_RE.sub(lambda m: 'neck.fpn_stages.%s.CSPStage_0.' % m.group(1), key)
    key = PAN_RE.sub(lambda m: 'neck.pan_stages.%s.CSPStage__0.' % m.group(1), key)
    key = CONVS_RE.sub(lambda m: '.convs.Conv_%s.' % m.group(1), key)
    key = key.replace('.convs.spp.', '.convs.SPP_0.')
    key = key.replace('._mean', '.running_mean').replace('._variance', '.running_var')
    return key


def convert(pd_state):
    out = {}
    # alpha scales the 1x1 branch of backbone RepVggBlocks; fold it into that
    # branch's BN params so the repo's alpha-free block is numerically equal
    alphas = {k[:-len('.alpha')]: v for k, v in pd_state.items() if k.endswith('.alpha')}
    for key, value in pd_state.items():
        if key.endswith('.alpha'):
            continue
        new_key = map_key(key)
        for prefix, alpha in alphas.items():
            if new_key.startswith(map_key(prefix) + '.conv2.'):
                if new_key.endswith('bn.weight') or new_key.endswith('bn.bias'):
                    value = value * alpha.item()
                break
        out[new_key] = torch.from_numpy(np.ascontiguousarray(value))
    for bn_key in [k for k in out if k.endswith('.bn.weight')]:
        out[bn_key[:-len('.weight')] + '.num_batches_tracked'] = torch.zeros((), dtype=torch.int64)
    return out


def main():
    parser = argparse.ArgumentParser(description='Convert PP-YOLOE .pdparams to PyTorch .pt')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    pd_state = load_pdparams(args.input)
    print(f'Loaded {len(pd_state)} Paddle parameters')
    state = convert(pd_state)
    print(f'Converted to {len(state)} PyTorch tensors')

    from models.yoloe import build_yoloe
    model = build_yoloe('s', num_classes=80)
    missing, unexpected = [], list(state.keys())
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        print('UNEXPECTED KEYS:', result.unexpected_keys[:10])
        sys.exit(1)
    if result.missing_keys:
        print('MISSING KEYS:', result.missing_keys[:10])
        sys.exit(1)
    print('Strict load check passed')

    torch.save(state, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
