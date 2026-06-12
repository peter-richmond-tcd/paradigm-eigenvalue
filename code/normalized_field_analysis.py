#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:23:41 2026

@author: peter
"""

"""
normalised_field_analysis.py
============================
Normalises λ₁^(field)(t) by paper count N^(field)(t) for each of the 12 fields
and 3 tiers, then plots on a common 2005=1 index scale.

Scientific question: Does the deep learning paradigm signal propagate from
computer science (Tier 1) into the physics-adjacent fields (Tiers 2 & 3),
and if so, with what time lag?

Reads:
  ~/Desktop/AI_cognition/paradigm_data/analysis/field_eigenvalues.csv
  ~/Desktop/AI_cognition/paradigm_data/analysis/tier_eigenvalues.csv

Writes:
  ~/Desktop/AI_cognition/paradigm_data/analysis/
    normalised_field_eigenvalues.csv
    normalised_tier_eigenvalues.csv
    diffusion_plots.pdf          ← main 4-panel figure
    diffusion_lag.csv            ← peak Δ(λ₁/N)/Δt year per field
    diffusion_lag_plot.pdf       ← lag bar chart
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────

ANALYSIS_DIR = Path.home() / "Desktop" / "AI_cognition" / "paradigm_data" / "analysis"

TIERS = {
    "computer_science":       1,
    "mathematics":            1,
    "condensed_matter":       1,
    "phase_transition":       2,
    "stochastic_process":     2,
    "dynamical_systems":      2,
    "complex_systems":        2,
    "statistical_mechanics":  3,
    "self_org_criticality":   3,
    "network_science":        3,
    "econophysics":           3,
    "information_theory":     3,
}
FIELDS = list(TIERS.keys())

# Display names (shorter for legends)
SHORT = {
    "computer_science":       "Computer sci.",
    "mathematics":            "Mathematics",
    "condensed_matter":       "Condensed matter",
    "phase_transition":       "Phase transition",
    "stochastic_process":     "Stochastic proc.",
    "dynamical_systems":      "Dynamical systems",
    "complex_systems":        "Complex systems",
    "statistical_mechanics":  "Stat. mechanics",
    "self_org_criticality":   "SOC",
    "network_science":        "Network science",
    "econophysics":           "Econophysics",
    "information_theory":     "Info. theory",
}

# Colour scheme: Tier 1 blues, Tier 2 browns/oranges, Tier 3 greens
COLOURS = {
    "computer_science":       "#0d2b6b",
    "mathematics":            "#3a6bb0",
    "condensed_matter":       "#7ab0d4",
    "phase_transition":       "#7b2d00",
    "stochastic_process":     "#b85c00",
    "dynamical_systems":      "#d48b3a",
    "complex_systems":        "#e8c07a",
    "statistical_mechanics":  "#1a5c1a",
    "self_org_criticality":   "#3d8c3d",
    "network_science":        "#70bb70",
    "econophysics":           "#a0d4a0",
    "information_theory":     "#c8eac8",
}
TIER_COLOURS = {1: "#0d2b6b", 2: "#7b2d00", 3: "#1a5c1a"}
TIER_LS      = {1: "-",       2: "--",      3: ":"}

BASE_YEAR = 2005

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Load and normalise ────────────────────────────────────────────────────────

def load_and_normalise_fields():
    path = ANALYSIS_DIR / "field_eigenvalues.csv"
    df = pd.read_csv(path)
    years = df["year"].values

    out = {"year": years}
    lag_records = []

    for field in FIELDS:
        lam_col = f"{field}_lambda1"
        n_col   = f"{field}_n"
        if lam_col not in df.columns:
            log.warning(f"Missing column {lam_col} — skipping")
            continue

        lam = df[lam_col].values.astype(float)
        n   = df[n_col].values.astype(float)

        # λ₁/N  (structural coherence per paper)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_per_n = np.where(n > 0, lam / n, np.nan)

        # Index: normalise so 2005 = 1
        base_mask = years == BASE_YEAR
        base_lam  = lam[base_mask][0]   if base_mask.any() else np.nan
        base_n    = n[base_mask][0]     if base_mask.any() else np.nan
        base_lpn  = lam_per_n[base_mask][0] if base_mask.any() else np.nan

        lam_idx  = lam / base_lam   if base_lam  > 0 else lam * np.nan
        n_idx    = n   / base_n     if base_n    > 0 else n   * np.nan
        lpn_idx  = lam_per_n / base_lpn if base_lpn > 0 else lam_per_n * np.nan

        out[f"{field}_lam_idx"]  = lam_idx
        out[f"{field}_n_idx"]    = n_idx
        out[f"{field}_lpn_idx"]  = lpn_idx
        out[f"{field}_lam_per_n"] = lam_per_n

        # Peak Δ(λ₁/N)/Δt — diffusion front year
        dlpn = np.gradient(np.nan_to_num(lpn_idx), years.astype(float))
        peak_yr = years[np.argmax(dlpn)]
        peak_val = dlpn.max()
        lag = int(peak_yr) - BASE_YEAR
        lag_records.append({
            "field":     field,
            "short":     SHORT[field],
            "tier":      TIERS[field],
            "peak_year": int(peak_yr),
            "lag_from_2005": lag,
            "peak_dlpn": float(peak_val),
        })
        log.info(f"  {SHORT[field]:22s}  peak Δ(λ₁/N)/Δt at {peak_yr}  "
                 f"(lag={lag:+d}yr, lpn_2023={lpn_idx[-1]:.2f})")

    norm_df  = pd.DataFrame(out)
    lag_df   = pd.DataFrame(lag_records).sort_values("peak_year")
    return norm_df, lag_df


def load_and_normalise_tiers():
    path = ANALYSIS_DIR / "tier_eigenvalues.csv"
    df   = pd.read_csv(path)
    years = df["year"].values

    out = {"year": years}
    for tier in [1, 2, 3]:
        lam_col = f"tier{tier}_lambda1"
        n_col   = f"tier{tier}_n"
        lam = df[lam_col].values.astype(float)
        n   = df[n_col].values.astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            lam_per_n = np.where(n > 0, lam / n, np.nan)

        base_mask = years == BASE_YEAR
        base_lpn  = lam_per_n[base_mask][0] if base_mask.any() else np.nan
        lpn_idx   = lam_per_n / base_lpn if base_lpn > 0 else lam_per_n * np.nan

        out[f"tier{tier}_lpn_idx"]  = lpn_idx
        out[f"tier{tier}_lam_per_n"] = lam_per_n

    return pd.DataFrame(out)


# ── Plotting ──────────────────────────────────────────────────────────────────

def shade(ax, years):
    ax.axvspan(2012, 2015, alpha=0.10, color="gold")
    ax.axvline(2012, color="goldenrod", lw=0.8, ls="--")
    ax.axvline(2015, color="goldenrod", lw=0.8, ls="--")
    ax.set_xlim(years[0] - 0.3, years[-1] + 0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Year", fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)


def plot_diffusion(norm_df, tier_df, lag_df):
    years = norm_df["year"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.38, wspace=0.30,
                        top=0.93, bottom=0.09, left=0.09, right=0.97)

    # ── Panel A: λ₁/N index, Tier 1 fields ───────────────────────────────
    ax = axes[0, 0]
    t1_fields = [f for f in FIELDS if TIERS[f] == 1]
    for field in t1_fields:
        col = f"{field}_lpn_idx"
        if col in norm_df.columns:
            ax.plot(years, norm_df[col], "o-",
                    color=COLOURS[field], lw=2.2, ms=5,
                    label=SHORT[field])
    shade(ax, years)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel(r"$\lambda_1/N$ index (2005 = 1)", fontsize=10)
    ax.set_title("A  Tier 1 — normalised structural coherence",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # ── Panel B: λ₁/N index, Tier 2 fields ───────────────────────────────
    ax = axes[0, 1]
    t2_fields = [f for f in FIELDS if TIERS[f] == 2]
    for field in t2_fields:
        col = f"{field}_lpn_idx"
        if col in norm_df.columns:
            ax.plot(years, norm_df[col], "o-",
                    color=COLOURS[field], lw=2, ms=5,
                    label=SHORT[field])
    shade(ax, years)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel(r"$\lambda_1/N$ index (2005 = 1)", fontsize=10)
    ax.set_title("B  Tier 2 — normalised structural coherence",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # ── Panel C: λ₁/N index, Tier 3 fields ───────────────────────────────
    ax = axes[1, 0]
    t3_fields = [f for f in FIELDS if TIERS[f] == 3]
    for field in t3_fields:
        col = f"{field}_lpn_idx"
        if col in norm_df.columns:
            ax.plot(years, norm_df[col], "o-",
                    color=COLOURS[field], lw=2, ms=5,
                    label=SHORT[field])
    shade(ax, years)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel(r"$\lambda_1/N$ index (2005 = 1)", fontsize=10)
    ax.set_title("C  Tier 3 — normalised structural coherence",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # ── Panel D: Tier-aggregated comparison on one plot ───────────────────
    ax = axes[1, 1]
    tier_years = tier_df["year"].values
    for tier in [1, 2, 3]:
        col = f"tier{tier}_lpn_idx"
        if col in tier_df.columns:
            ax.plot(tier_years, tier_df[col],
                    color=TIER_COLOURS[tier],
                    ls=TIER_LS[tier],
                    lw=2.5, ms=6, marker="o",
                    label=f"Tier {tier}")
    shade(ax, tier_years)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel(r"$\lambda_1/N$ index (2005 = 1)", fontsize=10)
    ax.set_title("D  All tiers — normalised comparison",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)

    plt.suptitle(r"Paradigm diffusion: normalised structural coherence $\lambda_1/N$ by field",
                 fontsize=13, y=0.975)
    out = ANALYSIS_DIR / "diffusion_plots.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"Saved → {out}")
    plt.close()


def plot_lag(lag_df):
    """
    Horizontal bar chart: peak Δ(λ₁/N)/Δt year for each field.
    Fields ordered by peak year; coloured by tier.
    A vertical line marks 2015 (end of AlexNet transition window).
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=0.26, right=0.95, top=0.91, bottom=0.10)

    df = lag_df.sort_values("peak_year", ascending=True).reset_index(drop=True)
    colours = [TIER_COLOURS[t] for t in df["tier"]]
    y = np.arange(len(df))

    bars = ax.barh(y, df["peak_year"] - BASE_YEAR,
                   left=BASE_YEAR, color=colours, alpha=0.80, height=0.6)

    # Annotate peak year on each bar
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["peak_year"] + 0.15, i, str(int(row["peak_year"])),
                va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["short"], fontsize=9)
    ax.axvline(2015, color="goldenrod", lw=1.5, ls="--", label="2015 (transition end)")
    ax.axvline(2012, color="goldenrod", lw=0.8, ls=":")
    ax.set_xlim(BASE_YEAR - 0.5, 2025)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.set_xlabel("Year of peak Δ(λ₁/N)/Δt  [diffusion front]", fontsize=10)
    ax.set_title("Paradigm diffusion lag — peak structural growth rate by field",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", ls=":", alpha=0.4)

    # Tier legend
    from matplotlib.patches import Patch
    legend_els = [Patch(color=TIER_COLOURS[t], label=f"Tier {t}") for t in [1, 2, 3]]
    legend_els.append(plt.Line2D([0], [0], color="goldenrod", ls="--",
                                 label="2015 (transition end)"))
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")

    out = ANALYSIS_DIR / "diffusion_lag_plot.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"Saved → {out}")
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(norm_df, lag_df):
    years = norm_df["year"].values
    log.info("\n── Normalised λ₁/N index — selected years ────────────────────")
    header = f"{'Field':25s}  {'Tier':4s}  " + \
             "  ".join(str(y) for y in [2005, 2010, 2012, 2015, 2017, 2019, 2021, 2023])
    log.info(header)
    log.info("-" * len(header))
    for field in FIELDS:
        col = f"{field}_lpn_idx"
        if col not in norm_df.columns:
            continue
        vals = norm_df[col].values
        sel_years = [2005, 2010, 2012, 2015, 2017, 2019, 2021, 2023]
        sel_vals  = [vals[list(years).index(y)] if y in years else np.nan
                     for y in sel_years]
        row = f"{SHORT[field]:25s}  T{TIERS[field]}    " + \
              "  ".join(f"{v:5.2f}" if not np.isnan(v) else "  --- " for v in sel_vals)
        log.info(row)

    log.info("\n── Diffusion lag table ───────────────────────────────────────")
    log.info(f"{'Field':25s}  {'Tier':4s}  {'Peak year':10s}  {'Lag (yr)':8s}")
    log.info("-" * 55)
    for _, row in lag_df.sort_values(["tier", "peak_year"]).iterrows():
        log.info(f"{row['short']:25s}  T{row['tier']}    "
                 f"{int(row['peak_year']):10d}  {int(row['lag_from_2005']):+8d}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Paradigm diffusion — normalised field eigenvalue analysis")
    log.info("=" * 60)

    norm_df, lag_df = load_and_normalise_fields()
    tier_df         = load_and_normalise_tiers()

    norm_df.to_csv(ANALYSIS_DIR / "normalised_field_eigenvalues.csv",
                   index=False, float_format="%.6f")
    lag_df.to_csv(ANALYSIS_DIR / "diffusion_lag.csv",
                  index=False, float_format="%.4f")
    tier_df.to_csv(ANALYSIS_DIR / "normalised_tier_eigenvalues.csv",
                   index=False, float_format="%.6f")

    print_summary(norm_df, lag_df)

    plot_diffusion(norm_df, tier_df, lag_df)
    plot_lag(lag_df)

    log.info("\nDone.")


if __name__ == "__main__":
    main()