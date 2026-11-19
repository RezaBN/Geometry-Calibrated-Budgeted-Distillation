"""Reference-equivalence tests for V3 NLP fast SWAT; run in NLP venv.

Run: PYTHONPATH=<v3-root> python -m unittest -v tests/test_nlp_fast_v3.py
"""
import unittest
import torch

from gcbd_v2.adapters.nlp import nlp_sample_geometry
from gcbd_v2.integrations.nlp import (
    cached_nlp_swat_loss_reference, cached_nlp_swat_loss,
    queried_nlp_final_swat_loss_reference, queried_nlp_final_swat_loss,
)


class TinyCache:
    def __init__(self, records):
        self.records = {i: item for i, item in enumerate(records)}
        self.manifest = {'completed_shards': {'0': {}}, 'total_bytes': 2048}
        self.max_open_shards = 4
        self.get_calls = 0
        self.get_many_calls = 0
    def get(self, i):
        self.get_calls += 1
        return self.records[int(i)]
    def get_many(self, ids):
        self.get_many_calls += 1
        return [self.records[int(i)] for i in ids]


def toy_inputs(batch=5, s_len=13, t_len=11):
    torch.manual_seed(79)
    def offsets_and_masks(length):
        offsets = torch.zeros(batch, length, 2, dtype=torch.long)
        mask = torch.zeros(batch, length, dtype=torch.long)
        for i in range(batch):
            # Intentional heterogeneous token counts and small support gaps.
            n = min(length - 2, 3 + 2 * i)
            for j in range(1, n + 1):
                offsets[i, j] = torch.tensor([j * 3 - 1, j * 3 + 1])
            mask[i, : n + 2] = 1
        lengths = torch.tensor([80 + 4*i for i in range(batch)])
        return offsets, mask, lengths
    so, sm, sl = offsets_and_masks(s_len)
    to, tm, tl = offsets_and_masks(t_len)
    return (so, sm, sl), (to, tm, tl)


class TestV3NLPFast(unittest.TestCase):
    def setUp(self):
        self.student, self.teacher = toy_inputs()
        b = self.student[0].shape[0]
        torch.manual_seed(19)
        self.student_states = [torch.randn(b, 13, 7) for _ in range(5)]
        self.teacher_states = [torch.randn(b, 11, 9) for _ in range(5)]
        self.records = nlp_sample_geometry(self.teacher_states, *self.teacher, [1, 2, 3])

    def test_cached_loss_and_gradient(self):
        for count in [0, 1, 2, 3]:
            for seed in [0, 991]:
                ref_states = [x.detach().clone().requires_grad_(True) for x in self.student_states]
                fast_states = [x.detach().clone().requires_grad_(True) for x in self.student_states]
                a = TinyCache(self.records)
                b = TinyCache(self.records)
                kw = dict(student_offsets=self.student[0],
                          student_attention_mask=self.student[1], text_lengths=self.student[2],
                          sample_ids=list(range(len(self.records))),
                          student_layer_indices=[1,2,3], teacher_layer_indices=[1,2,3],
                          active_layers=count, projections=4, seed=seed)
                old, _ = cached_nlp_swat_loss_reference(student_hidden_states=ref_states, cache=a, **kw)
                new, _ = cached_nlp_swat_loss(student_hidden_states=fast_states, cache=b, **kw)
                self.assertTrue(torch.allclose(new, old, rtol=2e-4, atol=1e-6),
                                f'cached loss mismatch {count=} {seed=}: {new} vs {old}')
                self.assertEqual(b.get_many_calls, 0 if count == 0 else 1)
                self.assertEqual(b.get_calls, 0)
                if count:
                    old.backward()
                    new.backward()
                    for sidx in [1,2,3][:count]:
                        self.assertTrue(torch.allclose(ref_states[sidx].grad, fast_states[sidx].grad,
                                                       rtol=4e-3, atol=2e-6),
                                        f'cached gradient mismatch {count=} {seed=} layer={sidx}')

    def test_queried_final_loss_and_gradient(self):
        for seed in [0, 991]:
            a = self.student_states[-1].detach().clone().requires_grad_(True)
            b = self.student_states[-1].detach().clone().requires_grad_(True)
            kw = dict(teacher_hidden_states=self.teacher_states,
                      student_offsets=self.student[0], student_attention_mask=self.student[1],
                      teacher_offsets=self.teacher[0], teacher_attention_mask=self.teacher[1],
                      text_lengths=self.student[2], teacher_text_lengths=self.teacher[2],
                      projections=4, seed=seed)
            old = queried_nlp_final_swat_loss_reference(student_final_hidden=a, **kw)
            new = queried_nlp_final_swat_loss(student_final_hidden=b, **kw)
            self.assertTrue(torch.allclose(new, old, rtol=2e-4, atol=1e-6),
                            f'queried loss mismatch {seed=}: {new} vs {old}')
            old.backward()
            new.backward()
            self.assertTrue(torch.allclose(a.grad, b.grad, rtol=4e-3, atol=2e-6),
                            f'queried gradient mismatch {seed=}')

    def test_real_provenance_locked_cache_reader(self):
        import tempfile
        from pathlib import Path
        from gcbd_v2.common.cache import GeometryCacheWriter, GeometryCacheReader
        metadata = {'purpose': 'v3-nlp-fast-unit-test', 'seed': 79}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            writer = GeometryCacheWriter(root, metadata)
            writer.write_shard(0, [0,1,2], self.records[:3])
            writer.write_shard(1, [3,4], self.records[3:])
            writer.finalize(range(5))
            reader = GeometryCacheReader(root, metadata)
            kwargs = dict(student_hidden_states=self.student_states,
                          student_offsets=self.student[0],
                          student_attention_mask=self.student[1],
                          text_lengths=self.student[2], sample_ids=[3,0,4,1,2],
                          student_layer_indices=[1,2,3], teacher_layer_indices=[1,2,3],
                          active_layers=3, projections=4, seed=991)
            # Sample orders differ from original student position; permute states
            # and token metadata to guarantee a correctly matched loss.
            order = torch.tensor([3,0,4,1,2])
            kwargs['student_hidden_states'] = [x[order] for x in self.student_states]
            kwargs['student_offsets'] = self.student[0][order]
            kwargs['student_attention_mask'] = self.student[1][order]
            kwargs['text_lengths'] = self.student[2][order]
            ref, _ = cached_nlp_swat_loss_reference(cache=reader, **kwargs)
            before = reader.io_metrics['shard_loads']
            fast, _ = cached_nlp_swat_loss(cache=reader, **kwargs)
            self.assertTrue(torch.allclose(ref, fast, rtol=2e-4, atol=1e-6))
            self.assertEqual(reader.io_metrics['shard_loads'], before)
            self.assertGreaterEqual(reader.max_open_shards, 2)

    def test_padded_supports_and_quantized_teacher(self):
        import gcbd_v2.integrations.nlp_fast as fast
        torch.manual_seed(111)
        ws = torch.rand(3, 9); ws[:, -4:] = 0
        ws = ws / ws.sum(-1, keepdim=True)
        wt = torch.rand(3, 7); wt[:, -2:] = 0
        wt = wt / wt.sum(-1, keepdim=True)
        ss = torch.rand(3, 9, 1)
        ts = torch.rand(3, 7, 1)
        # Exact reference per-row comparison, across projection signs.
        from gcbd_v2.common.geometry import swat_loss
        for seed in [0,1,991]:
            old = torch.stack([
                swat_loss(ws[i:i+1, :5], ss[i:i+1, :5],
                          wt[i:i+1, :5], ts[i:i+1, :5],
                          n_projections=8, seed=seed)
                for i in range(3)
            ]).mean()
            new = fast._swat_batched(ws, ss, wt, ts, projections=8, seed=seed)
            self.assertTrue(torch.allclose(old, new, rtol=2e-4, atol=2e-6),
                            f'padding mismatch {seed=}: {old} {new}')


if __name__ == '__main__':
    unittest.main()
