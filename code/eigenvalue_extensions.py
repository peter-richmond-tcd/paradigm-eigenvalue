#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:10:38 2026

@author: peter
"""

"""
eigenvalue_extensions.py
========================
Companion script to eigenvalue_analysis.py.  Three analyses:

  A) Normalised λ₁(t)/N(t) — separates structural change from volume growth
  B) Field-resolved λ₁^(field)(t) — tier-by-tier and individual field decomposition
  C) Top-concepts reader — tracks the leading eigenvector composition in
     computer_science year by year, revealing what the dominant mode *is*

Reads:
  ~/Desktop/AI_cognition/paradigm_data/processed/corpus_{year}_dedup.json
  ~/Desktop/AI_cognition/paradigm_data/analysis/concept_vocab.json   (from run A)

Writes to:
  ~/Desktop/AI_cognition/paradigm_data/analysis/
    normalised_lambda1.csv
    field_eigenvalues.csv
    cs_top_concepts_by_year.csv     ← concept name × year heat-map data
    extended_plots.pdf              ← 4-panel figure
    cs_concept_trajectories.pdf     ← concept weight trajectories

Usage:
    python eigenvalue_extensions.py
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR     = Path.home() / "Desktop" / "AI_cognition" / "paradigm_data" / "processed"
ANALYSIS_DIR = Path.home() / "Desktop" / "AI_cognition" / "paradigm_data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2005, 2024))   # stop at 2023 to avoid truncation artefact

# Field → tier mapping
TIERS = {
    "computer_science":    1,
    "mathematics":         1,
    "condensed_matter":    1,
    "phase_transition":    2,
    "stochastic_process":  2,
    "dynamical_systems":   2,
    "complex_systems":     2,
    "statistical_mechanics":  3,
    "self_org_criticality":   3,
    "network_science":        3,
    "econophysics":           3,
    "information_theory":     3,
}

FIELDS     = list(TIERS.keys())
ALL_TIERS  = [1, 2, 3]
N_EIGS     = 3
TOP_N      = 20    # top concepts to track per year in CS analysis

# Colours for fields and tiers
TIER_COLOURS = {1: "#1a3a6b", 2: "#8b4513", 3: "#2e6b2e"}
FIELD_COLOURS = {
    "computer_science":    "#1a3a6b",
    "mathematics":         "#3a6bb0",
    "condensed_matter":    "#7aadd4",
    "phase_transition":    "#8b4513",
    "stochastic_process":  "#c07840",
    "dynamical_systems":   "#d4a060",
    "complex_systems":     "#e8c890",
    "statistical_mechanics":  "#2e6b2e",
    "self_org_criticality":   "#559955",
    "network_science":        "#88cc88",
    "econophysics":           "#aaddaa",
    "information_theory":     "#cceecc",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Shared utilities ──────────────────────────────────────────────────────────

def load_vocab():
    path = ANALYSIS_DIR / "concept_vocab.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found at {path}.\n"
            "Run eigenvalue_analysis.py first to build concept_vocab.json."
        )
    with open(path) as f:
        vocab = json.load(f)
    log.info(f"Loaded vocabulary: {len(vocab):,} concepts")
    return vocab


def load_year(year):
    path = DATA_DIR / f"corpus_{year}_dedup.json"
    if not path.exists():
        log.warning(f"Missing: {path.name}")
        return []
    with open(path) as f:
        return json.load(f)


def build_cooccurrence(papers, vocab, field_filter=None):
    """
    Build symmetric sparse co-occurrence matrix.
    If field_filter is a set of field strings, only include papers whose
    source_field (or any entry in fields[]) is in that set.
    """
    V = len(vocab)
    rows, cols, data = [], [], []

    for paper in papers:
        # Field filtering
        if field_filter is not None:
            sf = paper.get("source_field", "")
            pf = set(paper.get("fields", []))
            if sf not in field_filter and not pf.intersection(field_filter):
                continue

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


def eigsh_top(C, k=N_EIGS):
    """Return (eigenvalues array length k, leading eigenvector length V)."""
    n = C.shape[0]
    k_actual = min(k, n - 2)
    if k_actual < 1 or C.nnz == 0:
        return np.zeros(k), np.zeros(n)
    norm = float(C.data.max())
    if norm == 0:
        return np.zeros(k), np.zeros(n)
    vals, vecs = spla.eigsh(C / norm, k=k_actual, which="LM", tol=1e-6, maxiter=5000)
    order = np.argsort(vals)[::-1]
    vals = vals[order] * norm
    vecs = vecs[:, order]
    if len(vals) < k:
        vals = np.concatenate([vals, np.zeros(k - len(vals))])
    return vals[:k], vecs[:, 0]


def count_papers(papers, field_filter=None):
    if field_filter is None:
        return len(papers)
    n = 0
    for p in papers:
        sf = p.get("source_field", "")
        pf = set(p.get("fields", []))
        if sf in field_filter or pf.intersection(field_filter):
            n += 1
    return n


# ── Analysis A: Normalised λ₁(t) ─────────────────────────────────────────────

def analysis_A(vocab):
    """Load pre-computed eigenvalue_results.csv and add normalised columns."""
    log.info("\n── Analysis A: Normalised λ₁(t) ──────────────────────────────")
    csv = ANALYSIS_DIR / "eigenvalue_results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Run eigenvalue_analysis.py first: {csv} not found.")
    df = pd.read_csv(csv)
    df = df[df["year"] <= 2023].copy()

    df["lambda1_per_paper"]  = df["lambda_1"] / df["n_papers"]
    df["lambda1_per_kpaper"] = df["lambda1_per_paper"] * 1000

    # Also compute relative growth index (2005 = 1)
    base = df.loc[df["year"] == 2005, "lambda_1"].values[0]
    base_n = df.loc[df["year"] == 2005, "n_papers"].values[0]
    df["lambda1_index"]         = df["lambda_1"] / base
    df["n_papers_index"]        = df["n_papers"] / base_n
    df["lambda1_norm_index"]    = df["lambda1_per_paper"] / (base / base_n)

    out = ANALYSIS_DIR / "normalised_lambda1.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    log.info(f"Saved → {out}")
    return df


# ── Analysis B: Field-resolved λ₁^(field)(t) ──────────────────────────────────

def analysis_B(vocab):
    """
    For each year, compute λ₁ for:
      - each individual field
      - each tier (union of fields in that tier)
    Returns two DataFrames: field_df, tier_df
    """
    log.info("\n── Analysis B: Field-resolved eigenvalues ─────────────────────")

    field_records = []
    tier_records  = []

    for year in YEARS:
        t0 = time.time()
        papers = load_year(year)
        if not papers:
            continue

        row_f = {"year": year}
        row_t = {"year": year}

        # Individual fields
        for field in FIELDS:
            ff = {field}
            C = build_cooccurrence(papers, vocab, field_filter=ff)
            eigs, _ = eigsh_top(C, k=1)
            n = count_papers(papers, field_filter=ff)
            row_f[f"{field}_lambda1"] = eigs[0]
            row_f[f"{field}_n"]       = n

        # Tiers
        for tier in ALL_TIERS:
            tier_fields = {f for f, t in TIERS.items() if t == tier}
            C = build_cooccurrence(papers, vocab, field_filter=tier_fields)
            eigs, _ = eigsh_top(C, k=1)
            n = count_papers(papers, field_filter=tier_fields)
            row_t[f"tier{tier}_lambda1"] = eigs[0]
            row_t[f"tier{tier}_n"]       = n

        elapsed = time.time() - t0
        log.info(f"  {year}: done ({elapsed:.1f}s)  "
                 f"CS λ₁={row_f['computer_science_lambda1']:.0f}  "
                 f"T1 λ₁={row_t['tier1_lambda1']:.0f}")

        field_records.append(row_f)
        tier_records.append(row_t)

    field_df = pd.DataFrame(field_records)
    tier_df  = pd.DataFrame(tier_records)

    field_df.to_csv(ANALYSIS_DIR / "field_eigenvalues.csv",
                    index=False, float_format="%.2f")
    tier_df.to_csv(ANALYSIS_DIR / "tier_eigenvalues.csv",
                   index=False, float_format="%.2f")
    log.info(f"Saved field and tier eigenvalue tables.")
    return field_df, tier_df


# ── Analysis C: CS top-concepts by year ───────────────────────────────────────

def analysis_C(vocab):
    """
    For each year, build the CS co-occurrence matrix, compute the leading
    eigenvector, and record the top-N concept weights.
    Returns a wide DataFrame (concepts × years) suitable for heat-map plotting.
    """
    log.info("\n── Analysis C: CS top-concepts by year ────────────────────────")

    idx_to_cid = {v: k for k, v in vocab.items()}
    concept_names = {}   # cid → name (populated on the fly)

    # year → {concept_name: weight}
    year_weights = {}

    for year in YEARS:
        t0 = time.time()
        papers = load_year(year)
        if not papers:
            continue

        # Collect names
        for p in papers:
            for c in p.get("concepts", []):
                cid = c.get("id")
                if cid and cid not in concept_names:
                    concept_names[cid] = c.get("name", cid)

        C = build_cooccurrence(papers, vocab, field_filter={"computer_science"})
        _, v1 = eigsh_top(C, k=1)

        weights = np.abs(v1)
        top_idx = np.argsort(weights)[::-1][:TOP_N]

        year_weights[year] = {
            concept_names.get(idx_to_cid[i], idx_to_cid[i]): float(weights[i])
            for i in top_idx
        }
        elapsed = time.time() - t0
        top3 = list(year_weights[year].keys())[:3]
        log.info(f"  {year}: top 3 = {top3}  ({elapsed:.1f}s)")

    # ── Build union of top concepts across all years ──────────────────────────
    # For the heat-map, keep concepts that appear in top-N in *any* year
    all_top = set()
    for yw in year_weights.values():
        all_top.update(yw.keys())

    # Score each concept by its mean weight across years (for ordering)
    mean_weight = {}
    for concept in all_top:
        mean_weight[concept] = np.mean([
            yw.get(concept, 0.0) for yw in year_weights.values()
        ])
    # Keep the top-30 most consistently important concepts
    top_concepts = sorted(mean_weight, key=mean_weight.get, reverse=True)[:30]

    # Build matrix: rows = concepts, cols = years
    heat = pd.DataFrame(
        index=top_concepts,
        columns=YEARS,
        dtype=float
    )
    for year in YEARS:
        yw = year_weights.get(year, {})
        for concept in top_concepts:
            heat.loc[concept, year] = yw.get(concept, 0.0)

    heat.to_csv(ANALYSIS_DIR / "cs_top_concepts_by_year.csv",
                float_format="%.6f")
    log.info(f"Saved cs_top_concepts_by_year.csv  "
             f"({len(top_concepts)} concepts × {len(YEARS)} years)")
    return heat, year_weights


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_extended(norm_df, field_df, tier_df, out_path):
    """4-panel figure: normalised λ₁, tier comparison, field comparison, spectral gap."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.32,
                        top=0.93, bottom=0.09, left=0.09, right=0.97)
    years = norm_df["year"].values

    def shade(ax):
        ax.axvspan(2012, 2015, alpha=0.10, color="gold")
        ax.axvline(2012, color="goldenrod", lw=0.7, ls="--")
        ax.axvline(2015, color="goldenrod", lw=0.7, ls="--")

    # ── Panel A: Normalised λ₁ and raw λ₁ overlaid ────────────────────────
    ax = axes[0, 0]
    ax2 = ax.twinx()
    l1, = ax.plot(years, norm_df["lambda1_index"], "o-",
                  color="#1a3a6b", lw=2.2, ms=5, label=r"$\lambda_1$ index")
    l2, = ax.plot(years, norm_df["n_papers_index"], "s--",
                  color="#aaaaaa", lw=1.5, ms=4, label="Paper count index")
    l3, = ax2.plot(years, norm_df["lambda1_norm_index"], "^-",
                   color="#c04040", lw=2, ms=5,
                   label=r"$\lambda_1/N$ index (structural)")
    shade(ax)
    ax.set_ylabel("Index (2005 = 1)", fontsize=10)
    ax2.set_ylabel(r"$\lambda_1/N$ index", fontsize=10, color="#c04040")
    ax2.tick_params(axis="y", colors="#c04040")
    ax.set_title("A  Volume vs structural growth", fontsize=11, fontweight="bold")
    lines = [l1, l2, l3]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)

    # ── Panel B: Tier-resolved λ₁(t) ──────────────────────────────────────
    ax = axes[0, 1]
    for tier in ALL_TIERS:
        col = f"tier{tier}_lambda1"
        if col in tier_df.columns:
            ax.plot(tier_df["year"], tier_df[col], "o-",
                    color=TIER_COLOURS[tier], lw=2, ms=5,
                    label=f"Tier {tier}")
    shade(ax)
    ax.set_ylabel(r"$\lambda_1$", fontsize=10)
    ax.set_title("B  Tier-resolved eigenvalues", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # ── Panel C: Individual field λ₁(t), Tier 1 fields highlighted ────────
    ax = axes[1, 0]
    for field in FIELDS:
        col = f"{field}_lambda1"
        if col not in field_df.columns:
            continue
        lw   = 2.2 if TIERS[field] == 1 else 1.0
        alpha = 1.0 if TIERS[field] == 1 else 0.55
        ls   = "-" if TIERS[field] == 1 else "--"
        label = field.replace("_", " ")
        ax.plot(field_df["year"], field_df[col],
                color=FIELD_COLOURS[field], lw=lw, ls=ls,
                alpha=alpha, label=label)
    shade(ax)
    ax.set_ylabel(r"$\lambda_1^{\rm field}$", fontsize=10)
    ax.set_title("C  Field-resolved eigenvalues", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # ── Panel D: Spectral gap λ₁/λ₂ for whole corpus + CS ─────────────────
    ax = axes[1, 1]
    ax.plot(years, norm_df["spectral_gap"], "o-",
            color="#1a3a6b", lw=2.2, ms=5, label="Full corpus")

    # CS spectral gap: need λ₂ from field_df if we stored it,
    # otherwise skip — field analysis only stored λ₁
    # (λ₂ for CS available from the global results for now)
    shade(ax)
    ax.set_ylabel(r"$\lambda_1 / \lambda_2$", fontsize=10)
    ax.set_title("D  Spectral gap (full corpus)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)

    for ax in axes.flat:
        ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlabel("Year", fontsize=9)

    plt.suptitle("Eigenvalue extensions — paradigm formation analysis",
                 fontsize=13, y=0.975)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info(f"Saved → {out_path}")


def plot_cs_concepts(heat, out_path):
    """
    Two sub-plots:
      (i)  Heat-map: concept weight (rows) × year (cols) — log colour scale
      (ii) Line trajectories for the top-10 concepts by peak weight
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 13))
    fig.subplots_adjust(hspace=0.40, top=0.94, bottom=0.07, left=0.24, right=0.97)

    years = [c for c in heat.columns]
    concepts = list(heat.index)
    mat = heat.values.astype(float)

    # ── Heat-map ──────────────────────────────────────────────────────────
    ax = axes[0]
    # Replace zeros with small value for log scale
    mat_plot = np.where(mat > 0, mat, mat[mat > 0].min() * 0.01
                        if (mat > 0).any() else 1e-10)
    im = ax.imshow(mat_plot, aspect="auto", cmap="YlOrRd",
                   norm=LogNorm(vmin=mat_plot.min(), vmax=mat_plot.max()),
                   interpolation="nearest")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=8)
    ax.axvline(years.index(2012) - 0.5, color="goldenrod", lw=1.5, ls="--")
    ax.axvline(years.index(2015) + 0.5, color="goldenrod", lw=1.5, ls="--")
    plt.colorbar(im, ax=ax, label="Eigenvector weight |v₁ᵢ|", pad=0.01)
    ax.set_title("A  Computer Science — leading eigenvector concept weights (heat-map)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)

    # ── Trajectory lines for top-10 concepts ─────────────────────────────
    ax2 = axes[1]
    peak_weights = mat.max(axis=1)
    top10_idx = np.argsort(peak_weights)[::-1][:10]
    cmap = plt.cm.tab10
    for rank, i in enumerate(top10_idx):
        ax2.plot(years, mat[i], "o-", color=cmap(rank), lw=1.8, ms=4,
                 label=concepts[i])
    ax2.axvspan(2012, 2015, alpha=0.10, color="gold")
    ax2.axvline(2012, color="goldenrod", lw=0.8, ls="--")
    ax2.axvline(2015, color="goldenrod", lw=0.8, ls="--")
    ax2.set_ylabel("Eigenvector weight |v₁ᵢ|", fontsize=10)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_title("B  Top-10 CS concepts — weight trajectories",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2, loc="upper left")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.set_xlim(years[0] - 0.5, years[-1] + 0.5)

    plt.suptitle("Computer Science — leading eigenvector composition by year",
                 fontsize=13, y=0.97)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Paradigm formation — eigenvalue extensions")
    log.info("=" * 60)

    vocab = load_vocab()

    # A: Normalised eigenvalues (fast — reads existing CSV)
    norm_df = analysis_A(vocab)

    # B: Field and tier decomposition (slow — re-reads corpus)
    field_df, tier_df = analysis_B(vocab)

    # C: CS top-concepts by year
    heat, year_weights = analysis_C(vocab)

    # Plots
    plot_extended(norm_df, field_df, tier_df,
                  ANALYSIS_DIR / "extended_plots.pdf")
    plot_cs_concepts(heat, ANALYSIS_DIR / "cs_concept_trajectories.pdf")

    # ── Print a readable top-concepts table for CS ────────────────────────
    log.info("\n── CS top-5 concepts by year ─────────────────────────────────")
    for year in YEARS:
        yw = year_weights.get(year, {})
        top5 = list(yw.items())[:5]
        names = ", ".join(f"{n} ({w:.4f})" for n, w in top5)
        log.info(f"  {year}: {names}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()