"""Synchronized reference/V3 SWAT forward+backward microbenchmark, not full training.

Examples:
  python tests/bench_nlp_fast_v3.py --device cpu --batch 16 --steps 2
  python tests/bench_nlp_fast_v3.py --device cuda --batch 32 --steps 3 \
      --report /scratch/.../reports/nlp/v3_smoke.json
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
import torch

from gcbd_v2.adapters.nlp import nlp_sample_geometry
from gcbd_v2.integrations.nlp import (
    cached_nlp_swat_loss_reference, cached_nlp_swat_loss,
    queried_nlp_final_swat_loss_reference, queried_nlp_final_swat_loss,
)


class TestCache:
    def __init__(self, records):
        self.records = records
        self.manifest = {'completed_shards': {'0': {}}, 'total_bytes': 1024}
        self.max_open_shards = 4
    def get(self, sample_id):
        return self.records[int(sample_id)]
    def get_many(self, sample_ids):
        return [self.records[int(i)] for i in sample_ids]


def offsets(batch, length, device, stride):
    os = torch.zeros(batch, length, 2, dtype=torch.long, device=device)
    mask = torch.zeros(batch, length, dtype=torch.long, device=device)
    for i in range(batch):
        n = max(3, min(length - 2, length - (i % 13) - 2))
        pos = torch.arange(1, n+1, device=device)
        os[i, 1:n+1, 0] = pos * stride - 1
        os[i, 1:n+1, 1] = pos * stride + 1
        mask[i, :n+2] = 1
    txt = torch.full((batch,), length * stride + 12, dtype=torch.long, device=device)
    return os, mask, txt


def build_fixture(device, batch):
    torch.manual_seed(2026)
    so, sm, sl = offsets(batch, 40, device, 3)
    to, tm, tl = offsets(batch, 48, device, 2)
    student = [torch.randn(batch, 40, 64, device=device) for _ in range(4)]
    teacher = [torch.randn(batch, 48, 96, device=device) for _ in range(4)]
    records = nlp_sample_geometry(teacher, to, tm, tl, [1, 2, 3])
    cache = TestCache(records)
    return student, teacher, cache, (so, sm, sl), (to, tm, tl)


def call(which, fn, student, teacher, cache, stokens, ttokens):
    if which == 'cached':
        return fn(student_hidden_states=student, student_offsets=stokens[0],
                  student_attention_mask=stokens[1], text_lengths=stokens[2],
                  sample_ids=list(range(student[0].shape[0])), cache=cache,
                  student_layer_indices=[1,2,3], teacher_layer_indices=[1,2,3],
                  active_layers=3, projections=8, seed=991)[0]
    return fn(student_final_hidden=student[-1], teacher_hidden_states=teacher,
              student_offsets=stokens[0], student_attention_mask=stokens[1],
              teacher_offsets=ttokens[0], teacher_attention_mask=ttokens[1],
              text_lengths=stokens[2], teacher_text_lengths=ttokens[2],
              projections=8, seed=991)


def benchmark(device, batch, steps):
    student, teacher, cache, stokens, ttokens = build_fixture(device, batch)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    output = {}
    for mode, fref, fnew in (
        ('cached', cached_nlp_swat_loss_reference, cached_nlp_swat_loss),
        ('queried', queried_nlp_final_swat_loss_reference, queried_nlp_final_swat_loss),
    ):
        timings = {}
        values = {}
        gradients = {}
        for label, fn in (('reference', fref), ('fast', fnew)):
            iterations = []
            for i in range(steps):
                s = [x.detach().clone().requires_grad_(True) for x in student]
                sync()
                start = time.perf_counter()
                loss = call(mode, fn, s, teacher, cache, stokens, ttokens)
                loss.backward()
                sync()
                iterations.append(time.perf_counter() - start)
                values[label] = loss.detach()
                gradients[label] = [x.grad.clone() if x.grad is not None else None for x in s]
            timings[label] = statistics.median(iterations)
        diff = (values['reference'] - values['fast']).abs().item()
        tolerance = 1e-6 + 2e-4 * abs(values['reference'].item())
        if diff > tolerance:
            raise AssertionError(f'{mode}: loss mismatch {diff} > {tolerance}')
        checked = [1,2,3] if mode == 'cached' else [3]
        max_grad_diff = 0.0
        for idx in checked:
            ga, gb = gradients['reference'][idx], gradients['fast'][idx]
            if not torch.allclose(ga, gb, rtol=4e-3, atol=2e-6):
                raise AssertionError(f'{mode}: gradient mismatch layer {idx}')
            max_grad_diff = max(max_grad_diff, float((ga-gb).abs().max().item()))
        output[mode] = {
            'reference_seconds_median': timings['reference'],
            'v3_seconds_median': timings['fast'],
            'speedup_x': timings['reference'] / timings['fast'],
            'loss_abs_error': diff,
            'gradient_max_abs_error': max_grad_diff,
            'numerical_check': 'PASS',
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1:
        parser.error('--steps and --batch must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    device = torch.device(args.device)
    source = Path(__file__).resolve().parents[1]
    core = ['gcbd_v2/integrations/nlp_fast.py', 'gcbd_v2/integrations/nlp.py',
            'nlp/GCBD-NLP_RUN.py']
    result = {
        'device': args.device,
        'device_name': torch.cuda.get_device_name(device) if args.device == 'cuda' else 'CPU',
        'torch_version': torch.__version__,
        'batch': args.batch,
        'steps_per_function': args.steps,
        'source_hashes': {f: hashlib.sha256((source/f).read_bytes()).hexdigest() for f in core},
        'results': benchmark(device, args.batch, args.steps),
    }
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
