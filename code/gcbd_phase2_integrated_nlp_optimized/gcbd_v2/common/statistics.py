"""Paired-seed inference helpers for final GCBD result tables."""
from __future__ import annotations
import numpy as np


def paired_seed_analysis(reference, proposed, confidence: float = 0.95,
                         n_boot: int = 20000, seed: int = 1234) -> dict:
    a, b = np.asarray(reference, dtype=float), np.asarray(proposed, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("Need paired one-dimensional seed arrays with n>=2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Seed arrays contain non-finite values")
    if not 0 < confidence < 1 or n_boot < 1000:
        raise ValueError("confidence must be in (0,1), n_boot must be >=1000")
    d = b - a
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[boot_idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    sd = float(d.std(ddof=1))
    try:
        from scipy.stats import ttest_rel, wilcoxon
        p_t = float(ttest_rel(b, a).pvalue)
        p_w = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
    except ImportError:
        p_t = p_w = None
    return {
        "n_pairs": len(d), "mean_difference": float(d.mean()), "sd_difference": sd,
        "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "bootstrap_ci": [float(lo), float(hi)], "paired_t_p": p_t,
        "wilcoxon_p": p_w, "seed_differences": d.tolist(),
    }


def noninferiority(reference, proposed, margin: float, confidence: float = 0.95) -> dict:
    from scipy.stats import t
    a, b = np.asarray(reference, dtype=float), np.asarray(proposed, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("Need paired one-dimensional seed arrays with n>=2")
    if margin <= 0 or not 0 < confidence < 1:
        raise ValueError("margin must be positive and confidence in (0,1)")
    d = b - a
    lower = float(d.mean() - t.ppf(confidence, len(d)-1) * d.std(ddof=1) / np.sqrt(len(d)))
    return {"margin": float(margin), "mean_difference": float(d.mean()),
            "one_sided_lower_bound": lower, "noninferior": bool(lower > -margin)}
