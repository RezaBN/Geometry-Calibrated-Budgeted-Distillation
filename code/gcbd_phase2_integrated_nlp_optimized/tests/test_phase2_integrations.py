import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from gcbd_v2.common.cache import GeometryCacheReader, GeometryCacheWriter
from gcbd_v2.common.geometry import swat_loss
from gcbd_v2.adapters.nlp import nlp_sample_geometry
from gcbd_v2.integrations.nlp import (
    GCBDClassificationCollator,
    GCBDLMTextCollator,
    build_nlp_geometry_cache,
    cached_nlp_swat_loss,
    differentiable_token_measure,
)
from gcbd_v2.integrations.vision import (
    IndexedDataset,
    build_vision_geometry_cache,
    cached_vision_swat_loss,
)
from gcbd_v2.integrations.transport import microbatched_transport_loss
from gcbd_v2.common.transport import semantic_cost


class DummyFastTokenizer:
    is_fast = True
    name_or_path = "dummy-fast"
    vocab_size = 32
    model_max_length = 128
    special_tokens_map = {"cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}
    padding_side = "right"
    truncation_side = "right"
    mask_token = "[MASK]"
    mask_token_id = 3

    def __len__(self):
        return self.vocab_size

    def get_special_tokens_mask(self, values, already_has_special_tokens=True):
        return [1 if int(x) in (1, 2) else 0 for x in values]

    def convert_tokens_to_ids(self, token):
        return self.mask_token_id

    def __call__(self, texts, padding=True, truncation=True, max_length=128,
                 return_offsets_mapping=True, return_tensors="pt"):
        rows, offsets = [], []
        for text in texts:
            # [CLS], one token per whitespace-delimited word, [SEP]
            words = []
            for i, word in enumerate(text.split()):
                start = text.find(word, 0 if i == 0 else text.find(words[-1][0]) + len(words[-1][0]))
                words.append((word, start, start + len(word)))
            ids = [1] + [4 + (len(w) % 20) for w, _, _ in words] + [2]
            off = [(0, 0)] + [(a, b) for _, a, b in words] + [(0, 0)]
            ids = ids[:max_length]
            off = off[:max_length]
            rows.append(ids)
            offsets.append(off)
        width = max(len(x) for x in rows)
        ids_t = torch.zeros((len(rows), width), dtype=torch.long)
        mask_t = torch.zeros_like(ids_t)
        offsets_t = torch.zeros((len(rows), width, 2), dtype=torch.long)
        for i, (row, off) in enumerate(zip(rows, offsets)):
            ids_t[i, :len(row)] = torch.tensor(row)
            mask_t[i, :len(row)] = 1
            offsets_t[i, :len(off)] = torch.tensor(off)
        return {"input_ids": ids_t, "attention_mask": mask_t, "offset_mapping": offsets_t}


class MiniDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row[item] for row in self.rows]
        return self.rows[item]

    def add_column(self, name, values):
        return MiniDataset([{**row, name: value} for row, value in zip(self.rows, values)])


class TinyTeacher(torch.nn.Module):
    def __init__(self, depth=3, vocab=32, dim=8, nclasses=2):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=depth,
            to_dict=lambda: {"num_hidden_layers": depth, "hidden_size": dim},
        )
        self.emb = torch.nn.Embedding(vocab, dim)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(depth)])
        self.head = torch.nn.Linear(dim, nclasses)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        x = self.emb(input_ids)
        hidden = [x]
        for layer in self.blocks:
            x = torch.tanh(layer(x))
            hidden.append(x)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1)
        return SimpleNamespace(hidden_states=tuple(hidden), logits=self.head(pooled))


def test_gcbd_collators_keep_offsets_and_separate_clean_lm_geometry():
    tok_t, tok_s = DummyFastTokenizer(), DummyFastTokenizer()
    cls = GCBDClassificationCollator(tok_t, tok_s, max_length=32)([
        {"text": "good movie", "label": 1, "sample_id": 7},
        {"text": "bad acting", "label": 0, "sample_id": 8},
    ])
    assert cls["sample_id"].tolist() == [7, 8]
    assert cls["student"]["offset_mapping"].shape[-1] == 2
    assert "offset_mapping" in cls["teacher"]

    lm = GCBDLMTextCollator(tok_t, tok_s, "masked", "masked", 32, 0.5)([
        {"text": "good movie", "sample_id": 7},
        {"text": "bad acting", "sample_id": 8},
    ])
    assert "labels" in lm["student"]
    assert "labels" not in lm["student_geometry"]
    assert torch.equal(lm["student_geometry"]["offset_mapping"], lm["student"]["offset_mapping"])


def test_differentiable_nlp_measure_preserves_student_gradient():
    hidden = torch.randn(2, 4, 6, requires_grad=True)
    offsets = torch.tensor([[[0, 0], [0, 4], [5, 9], [0, 0]], [[0, 0], [0, 3], [4, 8], [0, 0]]])
    mask = torch.ones(2, 4, dtype=torch.long)
    lengths = torch.tensor([9, 8])
    records = differentiable_token_measure(hidden, offsets, mask, lengths)
    target = [{"weights": torch.tensor([0.4, 0.6]), "support": torch.tensor([[0.2], [0.8]])} for _ in range(2)]
    loss = torch.stack([
        swat_loss(records[i]["weights"].unsqueeze(0), records[i]["support"].unsqueeze(0),
                  target[i]["weights"].unsqueeze(0), target[i]["support"].unsqueeze(0),
                  n_projections=4, quantile_bins=32)
        for i in range(2)
    ]).mean()
    loss.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_build_and_reuse_nlp_cache_then_student_swat_backward():
    tok = DummyFastTokenizer()
    dataset = MiniDataset([
        {"text": "good movie", "sample_id": 0},
        {"text": "bad acting", "sample_id": 1},
        {"text": "very good", "sample_id": 2},
    ])
    teacher = TinyTeacher(depth=3)
    with tempfile.TemporaryDirectory() as tmp:
        built = build_nlp_geometry_cache(
            teacher=teacher, dataset=dataset, tokenizer=tok, dataset_name="toy",
            scenario="toy-pair", seed=18, base_cache_dir=tmp, batch_size=2,
            max_length=16, device=torch.device("cpu"), num_workers=0,
        )
        assert built.ledger["geometry_examples"] == 3
        assert built.ledger["geometry_forward_invocations"] == 2
        assert built.reader.get(1)["layers"]
        reused = build_nlp_geometry_cache(
            teacher=teacher, dataset=dataset, tokenizer=tok, dataset_name="toy",
            scenario="toy-pair", seed=18, base_cache_dir=tmp, batch_size=2,
            max_length=16, device=torch.device("cpu"), num_workers=0,
        )
        assert reused.cache_hit
        student_hidden = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(4)]
        encoded = tok(["good movie", "bad acting"], return_tensors="pt")
        loss, _ = cached_nlp_swat_loss(
            student_hidden_states=student_hidden,
            student_offsets=encoded["offset_mapping"],
            student_attention_mask=encoded["attention_mask"],
            text_lengths=torch.tensor([10, 10]),
            sample_ids=torch.tensor([0, 1]),
            cache=built.reader,
            student_layer_indices=[1, 2],
            teacher_layer_indices=[1, 2],
            active_layers=2,
            projections=4,
        )
        loss.backward()
        assert student_hidden[1].grad is not None
        assert student_hidden[1].grad.abs().sum() > 0


class TinyImageDataset(Dataset):
    def __init__(self):
        self.data = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        self.targets = [0, 1, 0, 1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = torch.full((3, 8, 8), float(idx) / 4)
        return image, self.targets[idx]


class TinyVisionTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, images):
        features = self.conv(images)
        logits = self.head(features.mean((2, 3)))
        return logits, {"early": features}


def test_build_vision_cache_and_student_swat_backward():
    base = TinyImageDataset()
    indexed = IndexedDataset(base)
    teacher = TinyVisionTeacher()
    with tempfile.TemporaryDirectory() as tmp:
        built = build_vision_geometry_cache(
            teacher=teacher, indexed_dataset=indexed, dataset_name="CIFAR100-toy",
            scenario="A", seed=18, teacher_architecture="toy",
            teacher_layer_names=["early"], base_cache_dir=tmp,
            batch_size=2, device=torch.device("cpu"), num_workers=0,
            class_names=["zero", "one"],
        )
        assert built.ledger["geometry_examples"] == 4
        assert built.ledger["geometry_forward_invocations"] == 2
        shard_path = next(built.root.glob("shard_*.pt"))
        shard_payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        assert "early" in shard_payload["shared_supports"]
        assert "support_ref" in shard_payload["records"][0]["layers"]["early"]
        assert built.reader.get(0)["layers"]["early"]["support"].shape[-1] == 2
        student_feature = torch.randn(2, 4, 8, 8, requires_grad=True)
        loss, _ = cached_vision_swat_loss(
            student_features={"student_early": student_feature},
            sample_ids=torch.tensor([0, 1]), cache=built.reader,
            layer_pairs=[("early", "student_early")], projections=4,
        )
        loss.backward()
        assert student_feature.grad is not None and student_feature.grad.abs().sum() > 0
        assert built.reader.io_metrics["shard_loads"] >= 1


def test_semantic_transport_microbatching_has_student_gradient():
    embeddings = torch.randn(5, 7)
    cost = semantic_cost(embeddings)
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5)
    loss, diagnostics = microbatched_transport_loss(
        student, teacher, cost, mode="uot", iterations=20, microbatch_size=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert diagnostics["microbatches"] == 2


def test_shard_aware_batch_sampler_keeps_batches_cache_local_and_shuffles_epochs():
    from gcbd_v2.common.sampling import ShardAwareBatchSampler

    sampler = ShardAwareBatchSampler(dataset_size=11, batch_size=4, seed=123, drop_last=False)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    flat = [x for batch in epoch0 for x in batch]
    assert sorted(flat) == list(range(11))
    assert len(epoch0) == 3
    for batch in epoch0:
        assert len({x // 4 for x in batch}) == 1
    assert epoch0 != epoch1


def test_semantic_class_cost_requires_exact_label_order(tmp_path):
    from gcbd_v2.integrations.transport import load_semantic_cost

    path = tmp_path / "class_embeddings.pt"
    torch.save({
        "class_names": ["negative", "positive"],
        "class_texts": ["a negative review", "a positive review"],
        "embeddings": torch.randn(2, 4),
        "metadata": {"encoder": "local-test"},
    }, path)
    cost, metadata = load_semantic_cost(path, ["negative", "positive"], torch.device("cpu"))
    assert cost.shape == (2, 2)
    assert metadata["encoder"] == "local-test"
    with pytest.raises(ValueError, match="order"):
        load_semantic_cost(path, ["positive", "negative"], torch.device("cpu"))
