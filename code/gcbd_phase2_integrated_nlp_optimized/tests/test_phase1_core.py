import tempfile
import pytest
import torch

from gcbd_v2.common import (
    energy_measure, swat_loss, spatial_grid_support, token_offset_support,
    GeometryCacheWriter, GeometryCacheReader, CalibrationBuffer, cosine_frontier,
    select_gcbd, matched_rate_select, semantic_cost, entropic_transport_loss,
    TeacherLedger, linear_cka,
)
from gcbd_v2.adapters.nlp import layer_indices_by_depth, nlp_sample_geometry
from gcbd_v2.adapters.vision import vision_feature_measure, vision_sample_geometry


def test_energy_measure_mask_and_zero_features():
    x = torch.zeros(2, 4, 3, requires_grad=True)
    support = torch.linspace(0.125, 0.875, 4).view(4, 1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    w, r = energy_measure(x, support, mask)
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.equal(w[0, 2:], torch.zeros(2))
    assert torch.isfinite(r).all()
    with pytest.raises(ValueError):
        energy_measure(x, support, torch.zeros_like(mask))


def test_swat_heterogeneous_native_support_backward():
    s = torch.randn(2, 7, 8, requires_grad=True)
    t = torch.randn(2, 5, 12)
    ws, rs = energy_measure(s, torch.linspace(0.05, 0.95, 7).view(7, 1))
    wt, rt = energy_measure(t, torch.linspace(0.1, 0.9, 5).view(5, 1))
    loss = swat_loss(ws, rs, wt, rt, n_projections=5, quantile_bins=32)
    loss.backward()
    assert torch.isfinite(loss) and s.grad is not None and s.grad.abs().sum() > 0


def test_token_offsets_and_nlp_adapter():
    offsets = torch.tensor([[[0,0],[0,3],[4,7],[0,0],[0,0]], [[0,0],[0,2],[3,6],[0,0],[0,0]]])
    mask = torch.tensor([[1,1,1,1,0],[1,1,1,0,0]])
    supports, valid = token_offset_support(offsets, mask, torch.tensor([7,6]))
    assert valid.tolist() == [[False,True,True,False,False],[False,True,True,False,False]]
    assert supports.shape == (2,5,1)
    hs = [torch.randn(2,5,6), torch.randn(2,5,6), torch.randn(2,5,6), torch.randn(2,5,6)]
    records = nlp_sample_geometry(hs, offsets, mask, torch.tensor([7,6]), [1,2])
    assert len(records) == 2
    assert set(records[0]["layers"]) == {"1", "2"}
    assert torch.allclose(records[0]["layers"]["1"]["weights"].float().sum(), torch.tensor(1.0), atol=2e-3)
    assert layer_indices_by_depth(12) == [3,6,9]


def test_vision_cnn_and_vit_support():
    cnn = torch.randn(2, 16, 5, 7)
    w, r = vision_feature_measure(cnn)
    assert w.shape == (2,35) and r.shape == (2,35,2)
    vit = torch.randn(2, 197, 32)
    w2, r2 = vision_feature_measure(vit)
    assert w2.shape == (2,196) and r2.shape == (2,196,2)
    records = vision_sample_geometry({"l1": cnn, "l2": vit}, ["l1", "l2"])
    assert len(records) == 2 and set(records[0]["layers"]) == {"l1", "l2"}


def test_cache_provenance_coverage_and_reader():
    with tempfile.TemporaryDirectory() as d:
        metadata = {"dataset": "toy", "teacher_sha256": "abc", "view": "deterministic-v1"}
        writer = GeometryCacheWriter(d, metadata)
        records = [
            {"layers": {"1": {"weights": torch.tensor([0.4,0.6]), "support": torch.tensor([[0.25],[0.75]])}}},
            {"layers": {"1": {"weights": torch.tensor([0.2,0.8]), "support": torch.tensor([[0.25],[0.75]])}}},
        ]
        writer.write_shard(0, [10,11], records)
        writer.finalize([10,11])
        reader = GeometryCacheReader(d, metadata)
        assert torch.equal(reader.get(11)["layers"]["1"]["weights"], records[1]["layers"]["1"]["weights"])
        with pytest.raises(RuntimeError):
            GeometryCacheReader(d, {"dataset": "toy", "teacher_sha256": "wrong", "view": "deterministic-v1"})


def test_selection_budget_and_cosine_frontier():
    assert cosine_frontier(0, 5, 0.2, 1.0) == pytest.approx(0.2)
    assert cosine_frontier(4, 5, 0.2, 1.0) == pytest.approx(1.0)
    entropy = torch.linspace(0.01, 0.99, 100)
    buffer = CalibrationBuffer(256, min_values=16)
    selected, gate, weights, info = select_gcbd(entropy, 0.2, 0.8, buffer, rho=0.1)
    assert int(selected.sum()) <= 16
    assert gate.shape == weights.shape == selected.shape
    g = torch.Generator().manual_seed(7)
    assert int(matched_rate_select(entropy, int(selected.sum()), "random", g).sum()) == int(selected.sum())


def test_semantic_transport_and_gradient():
    embeddings = torch.randn(6, 10)
    cost = semantic_cost(embeddings)
    assert torch.allclose(cost, cost.T, atol=1e-6)
    assert torch.allclose(cost.diag(), torch.zeros(6), atol=1e-6)
    s = torch.randn(3, 6, requires_grad=True)
    t = torch.randn(3, 6)
    loss, diagnostics = entropic_transport_loss(s, t, cost, iterations=80, mode="uot")
    loss.backward()
    assert torch.isfinite(loss) and s.grad is not None
    assert diagnostics["finite"] and diagnostics["iterations"] >= 1


def test_ledger_and_cka():
    ledger = TeacherLedger()
    ledger.record_geometry(100, 4, 2.0)
    ledger.record_prediction(20, 2, 0.5)
    d = ledger.as_dict()
    assert d["teacher_examples_total"] == 120
    assert d["teacher_forward_invocations_total"] == 6
    x = torch.randn(40, 8)
    y = x @ torch.randn(8, 5)
    score = linear_cka(x, y)
    assert 0 <= float(score) <= 1.0001
