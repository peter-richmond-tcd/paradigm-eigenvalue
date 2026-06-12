#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:47:58 2026

@author: peter
"""

"""
eigenvalue_analysis.py
======================
Computes λ₁(t) — the leading eigenvalue of the concept co-occurrence matrix —
year by year across the deduplicated corpus, as a signature of the deep learning
paradigm transition (expected sharp rise ~2012–2015).

Author: Peter Richmond, Trinity College Dublin
Framework: Statistical physics of scientific paradigm formation

Usage (Spyder or terminal):
    python eigenvalue_analysis.py

Outputs (written to ANALYSIS_DIR):
    eigenvalue_results.csv       — λ₁(t), λ₂(t), λ₃(t), participation ratio, n_papers, n_concepts
    eigenvalue_plot.pdf          — publication-quality figure
    concept_vocab.json           — global concept vocabulary (id → index)
    top_concepts_{year}.json     — top-20 concepts by eigenvector weight, selected years
"""

import json
import os
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = Path.home() / "Desktop" / "AI_cognition" / "paradigm_data" / "processed"
ANALYSIS_DIR = Path.home() / "Desktop" / "AI_cognition" / "paradigm_data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

YEARS       = list(range(2005, 2026))          # 2005–2025 inclusive
N_EIGS      = 5                                 # number of leading eigenvalues to retain
MIN_CONCEPT_FREQ = 10                           # drop concepts appearing in < N papers/year globally
                                                # (applied per-year to filter noise)

# Tier labels for annotation
TIER_LABELS = {
    "computer_science":   "Tier 1",
    "mathematics":        "Tier 1",
    "condensed_matter":   "Tier 1",
    "phase_transition":   "Tier 2",
    "stochastic_process": "Tier 2",
    "dynamical_systems":  "Tier 2",
    "complex_systems":    "Tier 2",
    "statistical_mechanics":   "Tier 3",
    "self_org_criticality":    "Tier 3",
    "network_science":         "Tier 3",
    "econophysics":            "Tier 3",
    "information_theory":      "Tier 3",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Step 1: Build global concept vocabulary ───────────────────────────────────

def build_global_vocab(years, data_dir, min_freq=MIN_CONCEPT_FREQ):
    """
    Two-pass: count global concept frequencies across all years,
    then keep those above min_freq.  Returns {concept_id: index}.
    """
    log.info("Pass 1/2 — counting global concept frequencies …")
    freq = defaultdict(int)
    for year in years:
        path = data_dir / f"corpus_{year}_dedup.json"
        if not path.exists():
            log.warning(f"  Missing: {path.name} — skipping")
            continue
        with open(path) as f:
            papers = json.load(f)
        for paper in papers:
            seen = set()
            for c in paper.get("concepts", []):
                cid = c.get("id")
                if cid and cid not in seen:
                    freq[cid] += 1
                    seen.add(cid)
        log.info(f"  {year}: {len(papers):,} papers  (vocab so far: {len(freq):,})")

    log.info(f"Total unique concepts: {len(freq):,}")
    vocab = {cid: idx for idx, (cid, _) in
             enumerate(sorted((cid, f) for cid, f in freq.items() if f >= min_freq))}
    log.info(f"After min_freq={min_freq} filter: {len(vocab):,} concepts retained")
    return vocab


def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f)
    log.info(f"Vocabulary saved → {path}")


# ── Step 2: Co-occurrence matrix and eigenvalue decomposition ─────────────────

def build_cooccurrence_sparse(papers, vocab):
    """
    For each paper, collect the set of (filtered) concept indices that appear,
    then increment C[i,j] for every pair (i,j).  Returns a symmetric sparse
    matrix in CSR format.

    Memory strategy: accumulate in COO (row/col/data lists), convert once.
    For papers with k concepts the cost per paper is O(k²); k is typically
    small (median ~5–15 in OpenAlex), so this is fine.
    """
    V = len(vocab)
    rows, cols, data = [], [], []

    for paper in papers:
        idxs = sorted({vocab[c["id"]]
                       for c in paper.get("concepts", [])
                       if c.get("id") in vocab})
        if len(idxs) < 2:
            continue
        for ii, i in enumerate(idxs):
            for j in idxs[ii + 1:]:
                rows.append(i); cols.append(j); data.append(1)
                rows.append(j); cols.append(i); data.append(1)

    if not rows:
        return sp.csr_matrix((V, V), dtype=np.float32)

    C = sp.coo_matrix((data, (rows, cols)), shape=(V, V), dtype=np.float32)
    return C.tocsr()


def leading_eigenvalues(C, k=N_EIGS):
    """
    Compute k leading eigenvalues of symmetric sparse matrix C.
    Uses ARPACK (scipy's eigsh).  Returns sorted array, largest first.
    Also returns the leading eigenvector for concept interpretation.
    """
    n = C.shape[0]
    k_actual = min(k, n - 2)
    if k_actual < 1:
        return np.zeros(k), np.zeros(n)

    # Normalise by max element to improve numerical conditioning
    norm = C.data.max() if C.nnz > 0 else 1.0
    Cn = C / norm

    vals, vecs = spla.eigsh(Cn, k=k_actual, which="LM", tol=1e-6, maxiter=5000)
    order = np.argsort(vals)[::-1]
    vals = vals[order] * norm          # rescale back
    vecs = vecs[:, order]

    # Pad to length k if fewer eigenvalues computed
    if len(vals) < k:
        vals = np.concatenate([vals, np.zeros(k - len(vals))])
    return vals[:k], vecs[:, 0]


def participation_ratio(v):
    """
    PR = (Σᵢ vᵢ²)² / Σᵢ vᵢ⁴
    Measures effective number of concepts contributing to leading mode.
    PR → 1   : localised (one concept dominates)
    PR → N   : delocalised (all concepts contribute equally)
    """
    v2 = v ** 2
    s2 = v2.sum()
    if s2 == 0:
        return 0.0
    return float(s2 ** 2 / (v2 ** 2).sum())


# ── Step 3: Year-by-year loop ─────────────────────────────────────────────────

def analyse_corpus(years, data_dir, vocab):
    records = []
    concept_id_to_name = {}   # populated on the fly

    for year in years:
        t0 = time.time()
        path = data_dir / f"corpus_{year}_dedup.json"
        if not path.exists():
            log.warning(f"  {year}: file not found — skipped")
            continue

        with open(path) as f:
            papers = json.load(f)

        # Collect concept names while we have the data
        for paper in papers:
            for c in paper.get("concepts", []):
                cid = c.get("id")
                if cid and cid not in concept_id_to_name:
                    concept_id_to_name[cid] = c.get("name", cid)

        n_papers = len(papers)
        C = build_cooccurrence_sparse(papers, vocab)
        n_active = (np.diff(C.indptr) > 0).sum()   # concepts with ≥1 co-occurrence

        eigs, v1 = leading_eigenvalues(C, k=N_EIGS)
        pr = participation_ratio(v1)

        elapsed = time.time() - t0
        log.info(
            f"  {year}: {n_papers:>7,} papers | {n_active:>6,} active concepts | "
            f"λ₁={eigs[0]:.1f}  λ₂={eigs[1]:.1f}  PR={pr:.0f}  ({elapsed:.1f}s)"
        )

        rec = {"year": year, "n_papers": n_papers, "n_active_concepts": n_active,
               "participation_ratio": pr}
        for i, lam in enumerate(eigs):
            rec[f"lambda_{i+1}"] = lam
        records.append(rec)

        # Save top-20 concept weights for selected years
        if year in (2008, 2010, 2012, 2014, 2016, 2018, 2020, 2023, 2025):
            save_top_concepts(v1, vocab, concept_id_to_name, year)

    return pd.DataFrame(records), concept_id_to_name


def save_top_concepts(v1, vocab, names, year):
    """Save the 20 concepts with largest |v₁| weight for a given year."""
    idx_to_cid = {v: k for k, v in vocab.items()}
    weights = np.abs(v1)
    top_idx = np.argsort(weights)[::-1][:20]
    top = [{"rank": r + 1,
            "concept_id": idx_to_cid[i],
            "concept_name": names.get(idx_to_cid[i], "?"),
            "weight": float(weights[i])}
           for r, i in enumerate(top_idx)]
    out = ANALYSIS_DIR / f"top_concepts_{year}.json"
    with open(out, "w") as f:
        json.dump(top, f, indent=2)


# ── Step 4: Plot ──────────────────────────────────────────────────────────────

def plot_eigenvalues(df, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.subplots_adjust(hspace=0.08, top=0.93, bottom=0.08, left=0.12, right=0.96)

    years = df["year"].values
    lam1  = df["lambda_1"].values
    lam2  = df["lambda_2"].values
    lam3  = df["lambda_3"].values
    pr    = df["participation_ratio"].values
    n_pap = df["n_papers"].values / 1e3   # thousands

    # ── Panel A: λ₁(t) ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(years, lam1, "o-", color="#1a3a6b", lw=2.2, ms=6, label=r"$\lambda_1(t)$")
    ax.plot(years, lam2, "s--", color="#5588bb", lw=1.4, ms=4, alpha=0.7,
            label=r"$\lambda_2(t)$")
    ax.plot(years, lam3, "^:", color="#88bbdd", lw=1.2, ms=4, alpha=0.6,
            label=r"$\lambda_3(t)$")

    # Shade the transition window
    ax.axvspan(2012, 2015, alpha=0.12, color="gold", label="Transition window")
    ax.axvline(2012, color="goldenrod", lw=0.8, ls="--")
    ax.axvline(2015, color="goldenrod", lw=0.8, ls="--")

    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.set_title(r"Leading eigenvalue $\lambda_1(t)$ of concept co-occurrence matrix"
                 "\n(deep learning paradigm transition)", fontsize=12, pad=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", ls=":", alpha=0.4)

    # Annotate AlexNet
    if 2012 in list(years):
        y_alexnet = lam1[list(years).index(2012)]
        ax.annotate("AlexNet\n(2012)", xy=(2012, y_alexnet),
                    xytext=(2009.5, y_alexnet * 0.92),
                    arrowprops=dict(arrowstyle="->", color="#333"),
                    fontsize=8.5, color="#333")

    # ── Panel B: Participation ratio ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(years, pr, "o-", color="#8b1a1a", lw=2, ms=5)
    ax2.axvspan(2012, 2015, alpha=0.12, color="gold")
    ax2.set_ylabel("Participation\nratio PR", fontsize=11)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.grid(True, which="major", ls=":", alpha=0.4)

    # ── Panel C: Paper count ──────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.bar(years, n_pap, color="#4a7a4a", alpha=0.75, width=0.7)
    ax3.axvspan(2012, 2015, alpha=0.12, color="gold")
    ax3.set_ylabel("Papers (×10³)", fontsize=11)
    ax3.set_xlabel("Year", fontsize=12)
    ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax3.grid(True, which="major", axis="y", ls=":", alpha=0.4)

    for ax in axes:
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info(f"Plot saved → {out_path}")


# ── Step 5: Derivative analysis ───────────────────────────────────────────────

def derivative_analysis(df):
    """
    Compute Δλ₁/Δt (finite differences) and find the peak transition year.
    Prints a brief summary; appends dλ₁ column to df.
    """
    lam1 = df["lambda_1"].values.astype(float)
    dlam = np.gradient(lam1, df["year"].values.astype(float))
    df["d_lambda1_dt"] = dlam

    peak_idx = np.argmax(dlam)
    peak_year = df["year"].iloc[peak_idx]
    log.info(f"Peak Δλ₁/Δt at year {peak_year}  (value: {dlam[peak_idx]:.1f}/yr)")

    # Ratio λ₁/λ₂ — sharpening of spectral gap signals coherent mode emergence
    df["spectral_gap"] = df["lambda_1"] / df["lambda_2"].replace(0, np.nan)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Paradigm formation — eigenvalue analysis")
    log.info(f"Corpus: {DATA_DIR}")
    log.info(f"Years:  {YEARS[0]}–{YEARS[-1]}")
    log.info("=" * 60)

    vocab_path = ANALYSIS_DIR / "concept_vocab.json"
    if vocab_path.exists():
        log.info(f"Loading existing vocabulary from {vocab_path}")
        with open(vocab_path) as f:
            vocab = json.load(f)
        log.info(f"  {len(vocab):,} concepts")
    else:
        vocab = build_global_vocab(YEARS, DATA_DIR, min_freq=MIN_CONCEPT_FREQ)
        save_vocab(vocab, vocab_path)

    log.info("\nPass 2/2 — year-by-year eigenvalue analysis …")
    df, _ = analyse_corpus(YEARS, DATA_DIR, vocab)

    df = derivative_analysis(df)

    csv_path = ANALYSIS_DIR / "eigenvalue_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    log.info(f"Results saved → {csv_path}")

    plot_eigenvalues(df, ANALYSIS_DIR / "eigenvalue_plot.pdf")

    # Quick summary table
    log.info("\n── Summary ───────────────────────────────────────────────")
    cols = ["year", "n_papers", "lambda_1", "lambda_2", "spectral_gap", "participation_ratio"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    log.info("Done.")


if __name__ == "__main__":
    main()