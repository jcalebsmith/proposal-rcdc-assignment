#!/usr/bin/env python3
"""
Fetch gold standard data from NIH RePORTER API.
Downloads projects with official RCDC category assignments for validation.
"""

import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

os.chdir(Path(__file__).parent)

REPORTER_API = "https://api.reporter.nih.gov/v2/projects/search"

# Search terms to find diverse projects
SEARCH_TERMS = [
    "cancer treatment",
    "HIV vaccine",
    "diabetes prevention",
    "heart failure",
    "Alzheimer disease",
    "influenza virus",
    "depression treatment",
    "obesity intervention",
    "rare disease",
    "brain imaging",
    "lung cancer",
    "kidney transplant",
    "retinal degeneration",
    "clinical trial",
    "gene therapy",
    "genomic sequencing",
    "neural circuits",
    "biomaterials",
    "pediatric cancer",
    "aging brain",
    "maternal health",
    "health disparities",
    "vaccine development",
    "chronic pain",
    "opioid addiction",
    "immunotherapy",
    "stem cells",
    "machine learning health",
    "infectious disease",
    "tuberculosis",
    "malaria",
    "hepatitis",
    "stroke recovery",
    "Parkinson disease",
    "autism spectrum",
    "schizophrenia",
    "asthma",
    "arthritis",
    "osteoporosis",
    "sleep disorders",
]


def fetch_projects_by_text(search_text: str, limit: int = 25, offset: int = 0) -> list:
    """Fetch projects matching search text."""
    payload = {
        "criteria": {
            "advanced_text_search": {
                "operator": "and",
                "search_field": "all",
                "search_text": search_text,
            },
            "fiscal_years": [2024, 2023, 2022, 2021],
            "exclude_subprojects": True,
        },
        "offset": offset,
        "limit": limit,
        "sort_field": "award_notice_date",
        "sort_order": "desc",
    }

    try:
        resp = requests.post(REPORTER_API, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  Error: {e}")
        return []


def extract_project_data(proj: dict) -> dict:
    """Extract relevant fields from a project record."""
    # Get RCDC categories
    categories = proj.get("spending_categories_desc") or ""
    if isinstance(categories, list):
        categories = "; ".join(categories)

    # Get terms/keywords
    terms = proj.get("terms") or []
    if isinstance(terms, list):
        terms = "; ".join(terms)

    return {
        "AWARD_LONG_TITLE": proj.get("project_title", ""),
        "PRPSL_LONG_TITLE": proj.get("project_title", ""),
        "PRPSL_ABSTRACT": proj.get("abstract_text", "") or "",
        "KEY_TERMS": terms,
        "GOLD_STANDARD_CATEGORIES": categories,
        "project_num": proj.get("project_num", ""),
        "fiscal_year": proj.get("fiscal_year", ""),
        "has_abstract": bool(proj.get("abstract_text")),
    }


def main():
    all_projects = {}  # Use dict to dedupe by project_num

    print("Fetching projects from NIH RePORTER API...")
    print(f"Search terms: {len(SEARCH_TERMS)}")

    # Shuffle search terms to get variety
    search_terms = SEARCH_TERMS.copy()
    random.shuffle(search_terms)

    for term in search_terms:
        print(f"\nSearching: '{term}'")

        # Try multiple offsets to get more variety
        for offset in [0, 25, 50]:
            projects = fetch_projects_by_text(term, limit=25, offset=offset)

            new_count = 0
            for proj in projects:
                proj_num = proj.get("project_num")
                if proj_num and proj_num not in all_projects:
                    data = extract_project_data(proj)
                    # Must have categories and at least title
                    if data["GOLD_STANDARD_CATEGORIES"] and data["AWARD_LONG_TITLE"]:
                        all_projects[proj_num] = data
                        new_count += 1

            if new_count > 0:
                print(f"  offset={offset}: {new_count} new projects")

            time.sleep(0.3)  # Rate limiting

        # Stop if we have enough
        if len(all_projects) >= 500:
            print("\nReached target of 500 projects")
            break

    print(f"\n{'='*60}")
    print(f"Total unique projects: {len(all_projects)}")

    # Count projects with abstracts
    with_abstract = sum(1 for p in all_projects.values() if p["has_abstract"])
    print(f"Projects with abstracts: {with_abstract}")

    # Convert to DataFrame
    df = pd.DataFrame(list(all_projects.values()))

    # Remove helper columns for final output
    df_output = df.drop(columns=["project_num", "fiscal_year", "has_abstract"])

    # Save
    output_path = Path("data/nih_gold_standard_expanded.csv")
    df_output.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Also save raw JSON for reference
    json_path = Path("data/nih_gold_standard_expanded.json")
    with open(json_path, "w") as f:
        json.dump(list(all_projects.values()), f, indent=2)
    print(f"Saved raw data to: {json_path}")

    # Summary stats
    print(f"\n{'='*60}")
    print("TOP 30 CATEGORIES IN DATASET:")
    print("="*60)

    all_cats = {}
    for proj in all_projects.values():
        cats = proj["GOLD_STANDARD_CATEGORIES"].split("; ")
        for cat in cats:
            cat = cat.strip()
            if cat:
                all_cats[cat] = all_cats.get(cat, 0) + 1

    for cat, count in sorted(all_cats.items(), key=lambda x: -x[1])[:30]:
        print(f"  {cat}: {count}")

    print(f"\nTotal unique categories in dataset: {len(all_cats)}")


if __name__ == "__main__":
    main()
