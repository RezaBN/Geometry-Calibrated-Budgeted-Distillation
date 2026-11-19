#!/usr/bin/env python3
"""No-dataset numerical/gradient regression suite for Vision-only optimized SWAT.

Run on CPU or GPU. Fails before modifying any experiments if discrepancies are
above tolerance; original implementations are still available as references.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch
from gcbd_v2.adapters.vision import vision_feature_measure
from gcbd_v2.common.geometry import swat_loss
from gcbd_v2.integrations.vision import _single_swat
from gcbd_v2.integrations.vision_fast import (
    cached_vision_swat_loss_fast,
    queried_vision_final_swat_loss_fast,
    _batched_w2,
    _grid_plan,
    _cached_teacher_plan,
)

class FakeCache:
    def __init__(self, tensors):
        self.tensors = tensors
        self.max_open_shards = 4
        self.manifest = {'completed_shards': {'0': {'file': 'fake'}}}
        self.calls = 0
    def get_many(self, ids):
        self.calls += 1
        return [self.tensors[int(i)] for i in ids]


def assert_closeness(label, got, want, rtol=2e-3, atol=3e-5):
    a = float(got.detach().cpu()) if got.numel() == 1 else 'tensor'
    b = float(want.detach().cpu()) if want.numel() == 1 else 'tensor' 
    torch.testing.assert_close(got.detach(), want.detach(), rtol=rtol, atol=atol, msg=label)
    print(f'PASS {label}: optimized={a} reference={b}')


def test_final(features_s, features_t, *, projections=4, seed=991):
    ref_s = features_s.detach().clone().requires_grad_(True)
    new_s = features_s.detach().clone().requires_grad_(True)
    w_ref_s, s_ref_s = vision_feature_measure(ref_s)
    w_ref_t, s_ref_t = vision_feature_measure(features_t)
    orig_loss = swat_loss(w_ref_s, s_ref_s, w_ref_t, s_ref_t,
                          n_projections=projections, seed=seed)
    opt_loss = queried_vision_final_swat_loss_fast(new_s, features_t,
                                                    projections=projections, seed=seed)
    assert_closeness('FINAL forward', opt_loss, orig_loss)
    orig_grad = torch.autograd.grad(orig_loss, ref_s)[0]
    opt_grad = torch.autograd.grad(opt_loss, new_s)[0]
    assert_closeness('FINAL backward', opt_grad, orig_grad, rtol=1e-2, atol=1e-5)


def test_cached(s_features, t_features, *, projections=4, seed=0, fp16_support=True):
    student_ref = s_features.detach().clone().requires_grad_(True)
    student_new = s_features.detach().clone().requires_grad_(True)
    tw, ts = vision_feature_measure(t_features)
    cached = []
    for i in range(tw.shape[0]):
        cached.append({'layers': {'early': {
            'weights': tw[i].detach().half().cpu(),
            'support': ts[i].detach().to(torch.float16 if fp16_support else torch.float32).cpu(),
        }}})
    cache = FakeCache(cached)
    ids = torch.arange(tw.shape[0])
    ws, ss = vision_feature_measure(student_ref)
    ref_loss = torch.stack([
        _single_swat(ws[i], ss[i], cached[i]['layers']['early'], projections, seed)
        for i in range(tw.shape[0])
    ]).mean()
    fast_loss, diag = cached_vision_swat_loss_fast(
        student_features={'student_early': student_new},
        sample_ids=ids, cache=cache, layer_pairs=[('early','student_early')],
        active_layers=1, projections=projections, seed=seed)
    assert cache.calls == 1
    assert_closeness('CACHED forward', fast_loss, ref_loss)
    assert abs(diag['early_to_student_early'] - float(fast_loss.detach())) < 1e-8
    ref_grad = torch.autograd.grad(ref_loss, student_ref)[0]
    fast_grad = torch.autograd.grad(fast_loss, student_new)[0]
    assert_closeness('CACHED backward', fast_grad, ref_grad, rtol=2e-2, atol=1e-5)


def test_quantiles_sparse(device):
    # Sparse fp16 teacher with many zero-mass support entries and random weights.
    torch.manual_seed(101)
    s = torch.randn(3, 4, 5, 7, device=device)
    t = torch.randn(3, 8, 3, 6, device=device)
    ws, _ = vision_feature_measure(s)
    wt, ts = vision_feature_measure(t)
    wt = wt.clone()
    wt[:, ::3] = 0
    wt = wt / wt.sum(-1, keepdim=True)
    wt16 = wt.half().cpu()
    support16 = ts[0].half().cpu()
    wt_gpu = wt16.to(device).float()
    wt_gpu = wt_gpu / wt_gpu.sum(-1, keepdim=True)
    sp = _grid_plan(s, 4, 18)
    tp = _cached_teacher_plan(support16, 4, 18, device)
    fast = _batched_w2(ws, sp, wt_gpu, tp)
    grid_s = vision_feature_measure(s)[1]
    ref = swat_loss(ws, grid_s, wt_gpu,
                    support16.to(device).float().unsqueeze(0).expand(3,-1,-1),
                    n_projections=4, seed=18)
    assert_closeness('SPARSE zero-weights', fast, ref)


def test_shard_grouped_batch_reader():
    import tempfile
    from gcbd_v2.common.cache import GeometryCacheWriter, GeometryCacheReader
    with tempfile.TemporaryDirectory() as directory:
        metadata = {'name': 'swat-fast-reader-regression'}
        writer = GeometryCacheWriter(directory, metadata)
        for shard_id in range(3):
            ids = [shard_id * 2, shard_id * 2 + 1]
            records = [
                {'layers': {'early': {
                    'weights': torch.tensor([0.25, 0.75]),
                    'support': torch.tensor([[0.25, 0.25], [0.75, 0.75]]),
                }}, 'label': idx} for idx in ids
            ]
            writer.write_shard(shard_id, ids, records)
        writer.finalize(range(6))
        reader = GeometryCacheReader(directory, metadata, max_open_shards=1)
        requested = [5, 0, 4, 3, 1, 2, 5]
        grouped = reader.get_many(requested)
        assert [x['label'] for x in grouped] == requested
        assert reader.io_metrics['shard_loads'] == 3, reader.io_metrics
        for idx, record in zip(requested, grouped):
            reference = reader.get(idx)
            assert torch.equal(record['layers']['early']['weights'], reference['layers']['early']['weights'])
            assert torch.equal(record['layers']['early']['support'], reference['layers']['early']['support'])
    print('PASS grouped shard reads, original order, and support reconstruction')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU test requires CUDA allocation (srun or sbatch)')
    device = torch.device(args.device)
    torch.manual_seed(66)
    print('PyTorch:', torch.__version__, '| device:', device)
    test_shard_grouped_batch_reader()
    test_final(torch.randn(3, 5, 4, 6, device=device),
               torch.randn(3, 7, 2, 4, device=device))
    test_final(torch.randn(2, 9, 5, 3, device=device),
               torch.randn(2, 8, 4, 4, device=device), projections=6)
    # Heterogeneous native supports: ViT patch tokens vs CNN spatial map.
    test_final(torch.randn(2, 10, 8, device=device),
               torch.randn(2, 6, 2, 5, device=device), projections=5)
    test_cached(torch.randn(3, 7, 4, 4, device=device),
                torch.randn(3, 6, 2, 3, device=device))
    test_cached(torch.randn(3, 7, 4, 4, device=device),
                torch.randn(3, 6, 4, 4, device=device))
    test_quantiles_sparse(device)
    print('VISION SWAT LOSS + GRADIENT REGRESSION: PASS')

if __name__ == '__main__':
    main()
