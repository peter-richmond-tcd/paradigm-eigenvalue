#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:30:47 2026

@author: peter
"""

"""
physics_concept_trajectories.py
================================
For each of six target fields, builds the co-occurrence matrix year by year,
extracts the leading eigenvector, and tracks concept weight trajectories.

Scientific question: In physics-adjacent fields, are DL concepts rising toward
the *core* of the leading eigenvector (genuine paradigm diffusion) or remaining
peripheral while the eigenvector stays dominated by field-native concepts
(tool import / label arbitrage)?

Target fields:
  econophysics, stochastic_process, condensed_matter,
  statistical_mechanics, network_science, information_theory

Reads:
  ~/Desktop/AI_cognition/paradigm_data/processed/corpus_{year}_dedup.json
  ~/Desktop/AI_cognition/paradigm_data/analysis/concept_vocab.json

Writes (all to analysis/):
  physics_concept_data_{field}.csv    — concept × year weight matrices
  physics_trajectories.pdf            — 6-panel figure, one per field
  physics_dl_penetration.csv          — DL concept weight vs native concept
                                        weight by field by year
  physics_dl_penetration_plot.pdf     — the key diagnostic figure
"""

import json
import logging
import time
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

YEARS = list(range(2005, 2024))
TOP_N = 25   # top concepts to track per field

# Fields to analyse, with short display names
TARGET_FIELDS = {
    "econophysics":          "Econophysics",
    "stochastic_process":    "Stochastic processes",
    "condensed_matter":      "Condensed matter",
    "statistical_mechanics": "Statistical mechanics",
    "information_theory":    "Information theory",
    "network_science":       "Network science",
}

# DL/AI concept names to watch for (substring match, case-insensitive)
# These are the "invader" concepts from the DL paradigm
DL_KEYWORDS = [
    "deep learning",
    "machine learning",
    "neural network",
    "convolutional",
    "artificial intelligence",
    "transformer",
    "reinforcement learning",
    "recurrent",
    "generative",
    "random forest",
    "gradient boosting",
    "support vector",
    "natural language processing",
    "large language",
]

# Field-native concept keywords — words we expect to dominate in each field
# Used to compute the native vs invader balance
NATIVE_KEYWORDS = {
    "econophysics":          ["stock", "market", "financial", "price", "trading",
                              "portfolio", "volatility", "return", "econom", "wealth",
                              "agent", "power law", "fat tail", "herding"],
    "stochastic_process":    ["stochastic", "diffusion", "brownian", "markov",
                              "langevin", "fokker", "wiener", "martingale",
                              "random walk", "noise", "fluctuation", "process"],
    "condensed_matter":      ["spin", "lattice", "crystal", "fermi", "quantum",
                              "superconducti", "magnetic", "phonon", "electron",
                              "band", "topolog", "hall effect", "bose"],
    "statistical_mechanics": ["entropy", "partition", "boltzmann", "ising",
                              "phase transition", "free energy", "thermodynamic",
                              "canonical", "microstate", "ergodic", "renormali"],
    "information_theory":    ["entropy", "channel", "coding", "mutual information",
                              "compression", "shannon", "capacity", "rate",
                              "error", "bit", "signal"],
    "network_science":       ["network", "graph", "node", "edge", "degree",
                              "centrality", "clustering", "community",
                              "scale-free", "small world", "hub", "link"],
}

FIELD_COLOURS = {
    "econophysics":          "#7b2d00",
    "stochastic_process":    "#b85c00",
    "condensed_matter":      "#7ab0d4",
    "statistical_mechanics": "#1a5c1a",
    "information_theory":    "#c8eac8",
    "network_science":       "#3a6bb0",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Utilities (same pattern as previous scripts) ──────────────────────────────

def load_vocab():
    with open(ANALYSIS_DIR / "concept_vocab.json") as f:
        return json.load(f)

def load_year(year):
    path = DATA_DIR / f"corpus_{year}_dedup.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

def build_cooccurrence(papers, vocab, field_filter):
    V = len(vocab)
    rows, cols, data = [], [], []
    for paper in papers:
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

def eigsh_top(C, k=1):
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

def is_dl_concept(name):
    nl = name.lower()
    return any(kw in nl for kw in DL_KEYWORDS)

def is_native_concept(name, field):
    nl = name.lower()
    return any(kw in nl for kw in NATIVE_KEYWORDS.get(field, []))


# ── Per-field concept trajectory analysis ────────────────────────────────────

def analyse_field(field, vocab, all_papers_by_year):
    """
    For one field, across all years:
    - compute leading eigenvector
    - record top-N concept weights
    - separately sum weights of DL concepts vs native concepts

    Returns:
      heat  : DataFrame (top concepts × years)
      pen   : DataFrame (year, dl_weight_sum, native_weight_sum, dl_fraction)
    """
    log.info(f"\n  ── {TARGET_FIELDS[field]} ──────────────────────────")

    idx_to_cid   = {v: k for k, v in vocab.items()}
    concept_names = {}

    year_weights  = {}   # year → {name: weight}
    pen_records   = []

    for year in YEARS:
        papers = all_papers_by_year.get(year, [])
        if not papers:
            continue

        # Collect names
        for p in papers:
            for c in p.get("concepts", []):
                cid = c.get("id")
                if cid and cid not in concept_names:
                    concept_names[cid] = c.get("name", cid)

        C = build_cooccurrence(papers, vocab, field_filter={field})
        _, v1 = eigsh_top(C, k=1)

        weights = np.abs(v1)
        top_idx = np.argsort(weights)[::-1][:TOP_N]

        yw = {}
        dl_sum = 0.0
        native_sum = 0.0
        total_sum  = weights.sum()

        for i in range(len(weights)):
            cid  = idx_to_cid[i]
            name = concept_names.get(cid, cid)
            w    = float(weights[i])
            if w > 0:
                if is_dl_concept(name):
                    dl_sum += w
                if is_native_concept(name, field):
                    native_sum += w

        for i in top_idx:
            cid  = idx_to_cid[i]
            name = concept_names.get(cid, cid)
            yw[name] = float(weights[i])

        year_weights[year] = yw

        dl_frac = dl_sum / total_sum if total_sum > 0 else 0.0
        pen_records.append({
            "year": year, "field": field,
            "dl_weight_sum":     dl_sum,
            "native_weight_sum": native_sum,
            "total_weight_sum":  total_sum,
            "dl_fraction":       dl_frac,
            "native_fraction":   native_sum / total_sum if total_sum > 0 else 0.0,
            "dl_native_ratio":   dl_sum / native_sum if native_sum > 0 else np.nan,
        })

        top3 = list(yw.keys())[:3]
        log.info(f"    {year}: DL={dl_sum:.4f} ({dl_frac:.1%})  "
                 f"native={native_sum:.4f}  top3={top3}")

    # Build union of top concepts across years
    all_top = set()
    for yw in year_weights.values():
        all_top.update(yw.keys())

    mean_w = {c: np.mean([yw.get(c, 0.0) for yw in year_weights.values()])
              for c in all_top}
    top_concepts = sorted(mean_w, key=mean_w.get, reverse=True)[:TOP_N]

    heat = pd.DataFrame(index=top_concepts, columns=YEARS, dtype=float)
    for year in YEARS:
        yw = year_weights.get(year, {})
        for concept in top_concepts:
            heat.loc[concept, year] = yw.get(concept, 0.0)

    pen = pd.DataFrame(pen_records)
    return heat, pen


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectories(field_heats, field_pens):
    """
    6-panel figure: one panel per field.
    Each panel shows the top-8 concept weight trajectories,
    with DL concepts in red/warm tones and native in blue/cool tones.
    """
    fields = list(TARGET_FIELDS.keys())
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.subplots_adjust(hspace=0.42, wspace=0.28,
                        top=0.94, bottom=0.06, left=0.08, right=0.97)

    axes_flat = axes.flat
    for field, ax in zip(fields, axes_flat):
        heat = field_heats[field]
        years = [c for c in heat.columns if isinstance(c, int) or str(c).isdigit()]
        years_int = [int(y) for y in years]

        top8_concepts = list(heat.index[:8])
        cmap_dl     = plt.cm.Reds
        cmap_native = plt.cm.Blues
        dl_count     = sum(1 for c in top8_concepts if is_dl_concept(c))
        native_count = sum(1 for c in top8_concepts if is_native_concept(c, field))
        dl_i, nat_i = 0, 0

        for concept in top8_concepts:
            vals = heat.loc[concept, years].values.astype(float)
            if is_dl_concept(concept):
                colour = cmap_dl(0.4 + 0.5 * dl_i / max(dl_count, 1))
                lw, ls, ms = 2.2, "-", 6
                dl_i += 1
            elif is_native_concept(concept, field):
                colour = cmap_native(0.4 + 0.5 * nat_i / max(native_count, 1))
                lw, ls, ms = 2.0, "--", 5
                nat_i += 1
            else:
                colour = "grey"
                lw, ls, ms = 1.2, ":", 4

            label = concept[:30] + ("…" if len(concept) > 30 else "")
            ax.plot(years_int, vals, color=colour, lw=lw, ls=ls,
                    marker="o", ms=ms, alpha=0.85, label=label)

        ax.axvspan(2012, 2015, alpha=0.09, color="gold")
        ax.axvline(2012, color="goldenrod", lw=0.7, ls="--")
        ax.axvline(2015, color="goldenrod", lw=0.7, ls="--")
        ax.set_title(f"{TARGET_FIELDS[field]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Eigenvector weight |v₁ᵢ|", fontsize=9)
        ax.legend(fontsize=6.5, ncol=1, loc="upper left",
                  framealpha=0.85, handlelength=1.5)
        ax.grid(True, ls=":", alpha=0.35)
        ax.set_xlim(years_int[0] - 0.5, years_int[-1] + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    plt.suptitle("Physics-field concept trajectories: DL invasion vs native vocabulary\n"
                 "(red = DL/AI concepts, blue = field-native, grey = generic)",
                 fontsize=12, y=0.975)
    out = ANALYSIS_DIR / "physics_trajectories.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"\nSaved → {out}")
    plt.close()


def plot_dl_penetration(field_pens):
    """
    Key diagnostic: DL fraction of eigenvector weight by field by year.
    If DL fraction is rising but stays low → tool import.
    If DL fraction rises to dominate → genuine diffusion.
    Also plots DL/native ratio as the cleanest single diagnostic.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.09,
                        right=0.97, wspace=0.30)

    for field, pen in field_pens.items():
        years = pen["year"].values
        col   = FIELD_COLOURS[field]
        label = TARGET_FIELDS[field]

        # Panel A: DL fraction (proportion of total eigenvector weight)
        axes[0].plot(years, pen["dl_fraction"] * 100, "o-",
                     color=col, lw=2, ms=5, label=label)

        # Panel B: DL/native ratio (the core diagnostic)
        ratio = pen["dl_native_ratio"].replace([np.inf], np.nan)
        axes[1].plot(years, ratio, "o-",
                     color=col, lw=2, ms=5, label=label)

    for ax in axes:
        ax.axvspan(2012, 2015, alpha=0.09, color="gold")
        ax.axvline(2012, color="goldenrod", lw=0.8, ls="--")
        ax.axvline(2015, color="goldenrod", lw=0.8, ls="--")
        ax.set_xlabel("Year", fontsize=10)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.legend(fontsize=8.5, loc="upper left")

    axes[0].set_ylabel("DL concept weight as % of total eigenvector", fontsize=10)
    axes[0].set_title("A  DL penetration fraction", fontsize=11, fontweight="bold")

    axes[1].set_ylabel("DL weight sum / native weight sum", fontsize=10)
    axes[1].set_title("B  DL/native concept weight ratio\n"
                      "(ratio < 1 → field native; ratio > 1 → DL dominant)",
                      fontsize=11, fontweight="bold")
    axes[1].axhline(1.0, color="red", lw=1.2, ls="--", alpha=0.6,
                    label="DL = native (ratio = 1)")

    plt.suptitle("DL paradigm penetration into physics fields\n"
                 "Diffusion: DL fraction rises toward dominance  |  "
                 "Tool import: DL fraction rises but stays low",
                 fontsize=11, y=0.98)
    out = ANALYSIS_DIR / "physics_dl_penetration_plot.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"Saved → {out}")
    plt.close()


def plot_heatmaps(field_heats):
    """One heat-map per field, saved as a single multi-page-style tall figure."""
    fields = list(TARGET_FIELDS.keys())
    fig, axes = plt.subplots(3, 2, figsize=(16, 22))
    fig.subplots_adjust(hspace=0.35, wspace=0.35,
                        top=0.96, bottom=0.04, left=0.22, right=0.97)

    for field, ax in zip(fields, axes.flat):
        heat = field_heats[field]
        years = [int(c) for c in heat.columns]
        mat   = heat.values.astype(float)
        concepts = list(heat.index)

        # Colour DL concept labels red, native blue, generic black
        ycolours = []
        for c in concepts:
            if is_dl_concept(c):
                ycolours.append("#aa2222")
            elif is_native_concept(c, field):
                ycolours.append("#1a3a8b")
            else:
                ycolours.append("#333333")

        mat_plot = np.where(mat > 0, mat,
                            mat[mat > 0].min() * 0.01 if (mat > 0).any() else 1e-10)
        im = ax.imshow(mat_plot, aspect="auto", cmap="YlOrRd",
                       norm=LogNorm(vmin=mat_plot.min(), vmax=mat_plot.max()),
                       interpolation="nearest")

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(concepts)))
        ax.set_yticklabels(concepts, fontsize=7.5)
        for tick, colour in zip(ax.get_yticklabels(), ycolours):
            tick.set_color(colour)

        # Mark transition window
        try:
            x2012 = years.index(2012) - 0.5
            x2015 = years.index(2015) + 0.5
            ax.axvline(x2012, color="goldenrod", lw=1.5, ls="--")
            ax.axvline(x2015, color="goldenrod", lw=1.5, ls="--")
        except ValueError:
            pass

        plt.colorbar(im, ax=ax, label="|v₁ᵢ|", pad=0.02, shrink=0.8)
        ax.set_title(f"{TARGET_FIELDS[field]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)

    plt.suptitle("Physics field eigenvector heat-maps\n"
                 "Red labels = DL/AI concepts  |  Blue = field-native  |  Black = generic",
                 fontsize=12, y=0.975)
    out = ANALYSIS_DIR / "physics_heatmaps.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"Saved → {out}")
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_penetration_summary(field_pens):
    log.info("\n── DL penetration summary ────────────────────────────────────")
    log.info(f"{'Field':25s}  {'2010':>6}  {'2014':>6}  {'2017':>6}  "
             f"{'2020':>6}  {'2023':>6}  {'ratio_2023':>10}")
    log.info("-" * 80)
    for field, pen in field_pens.items():
        def get(col, yr):
            row = pen[pen["year"] == yr]
            return row[col].values[0] if len(row) else np.nan
        frac = lambda yr: get("dl_fraction", yr)
        ratio = lambda yr: get("dl_native_ratio", yr)
        log.info(
            f"{TARGET_FIELDS[field]:25s}  "
            f"{frac(2010)*100:5.1f}%  {frac(2014)*100:5.1f}%  "
            f"{frac(2017)*100:5.1f}%  {frac(2020)*100:5.1f}%  "
            f"{frac(2023)*100:5.1f}%  {ratio(2023):>10.3f}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Physics field concept trajectory analysis")
    log.info("=" * 60)

    vocab = load_vocab()

    # Load corpus once, cache by year — avoids re-reading disk 6× per year
    log.info("Loading corpus into memory …")
    all_papers_by_year = {}
    for year in YEARS:
        papers = load_year(year)
        all_papers_by_year[year] = papers
        log.info(f"  {year}: {len(papers):,} papers")

    field_heats = {}
    field_pens  = {}

    for field in TARGET_FIELDS:
        heat, pen = analyse_field(field, vocab, all_papers_by_year)
        field_heats[field] = heat
        field_pens[field]  = pen
        heat.to_csv(ANALYSIS_DIR / f"physics_concept_data_{field}.csv",
                    float_format="%.6f")
        pen.to_csv(ANALYSIS_DIR / f"physics_penetration_{field}.csv",
                   index=False, float_format="%.6f")

    # Combine penetration data
    all_pen = pd.concat(field_pens.values(), ignore_index=True)
    all_pen.to_csv(ANALYSIS_DIR / "physics_dl_penetration.csv",
                   index=False, float_format="%.6f")

    print_penetration_summary(field_pens)

    log.info("\nGenerating plots …")
    plot_trajectories(field_heats, field_pens)
    plot_dl_penetration(field_pens)
    plot_heatmaps(field_heats)

    log.info("\nDone.")


if __name__ == "__main__":
    main()