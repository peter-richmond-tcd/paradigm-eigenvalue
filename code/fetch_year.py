#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_year.py
=============
Fetches the OpenAlex bibliometric corpus for a single target year
across all twelve fields used in the paradigm formation analysis.

Usage:
    Set TARGET_YEAR below to the desired year (2005–2023) and run:

        python fetch_year.py

    Repeat for each year in sequence. Each run completes in 1–3 hours
    depending on the year, well within OpenAlex rate limit tolerances.
    Files already successfully downloaded are skipped automatically,
    so the script can be safely interrupted and restarted.

Output:
    One JSON file per field per year, written to:
        ~/Desktop/AI_cognition/paradigm_data/raw/
    Filename format: {field_name}_{year}.json
    Each file contains a list of paper records with keys:
        id, title, abstract, year, cited_by, concepts, source_field

Configuration:
    TARGET_YEAR   — set this before each run
    EMAIL         — your email address for the OpenAlex polite pool
    RAW_DIR       — output directory (change if needed)

Peter Richmond, Trinity College Dublin
For: "Spectral signatures of scientific paradigm formation:
      an eigenvalue analysis of concept co-occurrence networks"
Physica A: Statistical Mechanics and its Applications (2025)
"""

import requests, json, time
from pathlib import Path

# ── CONFIGURATION — edit these before running ─────────────────────────────────

TARGET_YEAR = 2005        # ← set this to the year you want to fetch
                          #   run once for each year from 2005 to 2023

EMAIL = "peter_richmond@icloud.com"   # ← your email (enables polite pool)

RAW_DIR = Path.home() / "Desktop" / "AI_cognition" / "paradigm_data" / "raw"

# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR.mkdir(parents=True, exist_ok=True)

# OpenAlex concept IDs for the twelve fields
FIELDS = {
    "computer_science":      "C41008148",
    "mathematics":           "C33923547",
    "condensed_matter":      "C26873012",
    "phase_transition":      "C149288129",
    "stochastic_process":    "C8272713",
    "dynamical_systems":     "C79379906",
    "complex_systems":       "C47822265",
    "statistical_mechanics": "C99874945",
    "self_org_criticality":  "C103200210",
    "network_science":       "C137753397",
    "econophysics":          "C29912722",
    "information_theory":    "C52622258",
}

# Filtering thresholds
MIN_CITATIONS     = 3      # minimum citation count for inclusion
MIN_CONCEPT_SCORE = 0.5    # minimum OpenAlex concept relevance score
PRIMARY_SCORE     = 0.6    # minimum score for the field's primary concept

# Rate limiting
PAGE_DELAY  = 1.5    # seconds between paginated API requests
FIELD_DELAY = 120    # seconds between fields (avoids rate limiting)


def reconstruct_abstract(inv):
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inv:
        return ""
    pos = []
    for word, locs in inv.items():
        for p in locs:
            pos.append((p, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def passes_filters(work, field_id):
    """Return True if the paper's primary field concept meets PRIMARY_SCORE."""
    for c in work.get("concepts", []):
        if (c.get("id", "").split("/")[-1] == field_id
                and c.get("score", 0) >= PRIMARY_SCORE):
            return True
    return False


def is_valid(fpath):
    """Return True if a JSON file exists, is non-empty, and parses correctly."""
    if not fpath.exists() or fpath.stat().st_size < 10:
        return False
    try:
        d = json.loads(fpath.read_text())
        return isinstance(d, list) and len(d) > 0
    except Exception:
        return False


def fetch_one(field_name, field_id, year):
    """
    Fetch all papers for one field and one year from the OpenAlex API.
    Uses cursor-based pagination. Retries on HTTP errors and rate limits.
    Returns a list of paper dicts.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": f"mailto:{EMAIL}",
        "Accept":     "application/json",
    })
    params = {
        "filter"  : (f"concepts.id:{field_id},"
                     f"publication_year:{year},"
                     f"has_abstract:true,"
                     f"cited_by_count:>{MIN_CITATIONS}"),
        "per-page": 200,
        "cursor"  : "*",
        "select"  : ("id,title,abstract_inverted_index,"
                     "publication_year,concepts,cited_by_count"),
    }
    papers, page, seen, retries = [], 0, 0, 0

    while True:
        try:
            r = session.get("https://api.openalex.org/works",
                            params=params, timeout=30)

            if r.status_code == 429:
                wait = min(300, 60 * (2 ** retries))
                print(f"\n    429 rate limit — waiting {wait}s", flush=True)
                time.sleep(wait)
                retries += 1
                if retries > 6:
                    print("    Too many retries — will resume on next run")
                    break
                continue

            if r.status_code != 200:
                print(f"\n    HTTP {r.status_code} — retrying in 30s")
                time.sleep(30)
                retries += 1
                if retries > 4:
                    break
                continue

            data    = r.json()
            results = data.get("results", [])
            seen   += len(results)
            retries = 0

            for work in results:
                if not passes_filters(work, field_id):
                    continue
                concepts = [
                    {"id":    c["id"].split("/")[-1],
                     "name":  c["display_name"],
                     "score": c["score"]}
                    for c in work.get("concepts", [])
                    if c["score"] >= MIN_CONCEPT_SCORE
                ]
                papers.append({
                    "id":           work.get("id", "").split("/")[-1],
                    "title":        work.get("title", ""),
                    "abstract":     reconstruct_abstract(
                                        work.get("abstract_inverted_index")),
                    "year":         work.get("publication_year"),
                    "cited_by":     work.get("cited_by_count", 0),
                    "concepts":     concepts,
                    "source_field": field_name,
                })

            page += 1
            print(f"    page {page:3d} | seen {seen:7,} | kept {len(papers):6,}",
                  end="\r", flush=True)

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            params["cursor"] = cursor
            time.sleep(PAGE_DELAY)

        except Exception as e:
            print(f"\n    Error: {e} — retrying in 30s")
            time.sleep(30)
            retries += 1
            if retries > 4:
                break

    print(flush=True)
    return papers


def main():
    year = TARGET_YEAR
    print(f"\n{'='*55}")
    print(f"fetch_year.py — OpenAlex corpus retrieval")
    print(f"Target year : {year}")
    print(f"Email       : {EMAIL}")
    print(f"Output dir  : {RAW_DIR}")
    print(f"Page delay  : {PAGE_DELAY}s  |  Field delay: {FIELD_DELAY}s")
    print(f"{'='*55}\n")

    todo = [(fn, fid) for fn, fid in FIELDS.items()
            if not is_valid(RAW_DIR / f"{fn}_{year}.json")]
    done = [fn for fn in FIELDS
            if is_valid(RAW_DIR / f"{fn}_{year}.json")]

    if done:
        print(f"Already complete ({len(done)} fields):")
        for fn in done:
            fpath = RAW_DIR / f"{fn}_{year}.json"
            d = json.loads(fpath.read_text())
            print(f"  {fn:25s}: {len(d):,} papers")

    print(f"\nTo fetch: {len(todo)} fields\n")

    for i, (field_name, field_id) in enumerate(todo, 1):
        fpath = RAW_DIR / f"{field_name}_{year}.json"
        print(f"[{i}/{len(todo)}]  {field_name:25s}  year {year}", flush=True)

        papers = fetch_one(field_name, field_id, year)

        # Write atomically via a temporary file
        tmp = fpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(papers, ensure_ascii=False))
        tmp.rename(fpath)
        print(f"    ✓  {len(papers):,} papers  "
              f"({fpath.stat().st_size // 1024} KB)\n")

        if i < len(todo):
            print(f"    Pausing {FIELD_DELAY}s before next field ...\n",
                  flush=True)
            time.sleep(FIELD_DELAY)

    print(f"{'='*55}")
    print(f"Year {year} complete.")
    print(f"{'='*55}\n")
    if year < 2023:
        print(f"Next step: set TARGET_YEAR = {year + 1} and run again.\n")
    else:
        print("All years complete. Proceed to eigenvalue_analysis.py.\n")


if __name__ == "__main__":
    main()
