#!/usr/bin/env python3
"""Controlled small-batch GPU benchmark: original and vectorized cached SWAT."""
import argparse
import time
import torch
from gcbd_v2.adapters.vision import vision_feature_measure
from gcbd_v2.common.geometry import swat_loss
from gcbd_v2.integrations.vision import cached_vision_swat_loss
from gcbd_v2.integrations.vision import _single_swat

class MiniCache:
    def __init__(self, records):
        self.records = records
        self.max_open_shards = 4
        self.manifest = {'completed_shards': {'0': {}}}
    def get_many(self, ids):
        return [self.records[int(i)] for i in ids]

def time_one(fn, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    return result, elapsed, peak

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
p.add_argument('--images', type=int, default=8)
p.add_argument('--side', type=int, default=56)
p.add_argument('--projections', type=int, default=8)
a = p.parse_args()
assert a.images >= 1 and a.side >= 2 and a.projections >= 1
if a.device == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('GPU benchmark requires allocated GPU')
device = torch.device(a.device)
torch.manual_seed(33)
feat = torch.randn(a.images, 8, a.side, a.side, device=device)
teach = torch.randn(a.images, 8, a.side, a.side, device=device)
wt, st = vision_feature_measure(teach)
records = [{'layers': {'early': {'weights': wt[i].half().cpu(),
                                 'support': st[i].half().cpu()}}} for i in range(a.images)]
cache = MiniCache(records)

def reference():
    x = feat.detach().clone().requires_grad_(True)
    ws, ss = vision_feature_measure(x)
    value = torch.stack([_single_swat(ws[i], ss[i], records[i]['layers']['early'],
                                      a.projections, 0) for i in range(a.images)]).mean()
    value.backward()
    return value.detach(), x.grad.detach()

def optimized():
    x = feat.detach().clone().requires_grad_(True)
    value, _ = cached_vision_swat_loss(student_features={'s': x},
                                       sample_ids=list(range(a.images)), cache=cache,
                                       layer_pairs=[('early', 's')],
                                       projections=a.projections)
    value.backward()
    return value.detach(), x.grad.detach()

(ref, grad_ref), t_ref, memory_ref = time_one(reference, device)
(opt, grad_opt), t_fast, memory_fast = time_one(optimized, device)
torch.testing.assert_close(ref, opt, rtol=2e-3, atol=3e-5)
torch.testing.assert_close(grad_ref, grad_opt, rtol=2e-2, atol=1e-5)
print('Device:', device, '| batch:', a.images, '| support:', a.side, 'x', a.side,
      '| directions:', a.projections)
print(f'Reference forward+backward: {t_ref:.4f} sec')
print(f'Optimized forward+backward: {t_fast:.4f} sec')
print(f'Speed ratio on this synthetic minibatch: {t_ref/max(t_fast,1e-8):.2f}x')
if memory_ref is not None:
    print(f'Peak torch allocated reference: {memory_ref/1024**2:.1f} MiB')
    print(f'Peak torch allocated optimized: {memory_fast/1024**2:.1f} MiB')
print('H100 SWAT BENCHMARK + GRADIENT CHECK: PASS')
