#!/usr/bin/env python3
"""
rcdc_classifier.py

Classify award rows into NIH RCDC categories using the latest NIH-published
(public) RCDC category definition summaries.

Key features
- Always fetches the current NIH RCDC category definitions (with local caching + revalidation).
- Builds a text representation of each award from configured fields.
- Uses TF-IDF + cosine similarity for classification.
- Multi-label output: returns multiple categories above a confidence threshold.
- Rule-based keyword matching: guarantees certain categories when keywords are present.
- Query expansion: expands award text with MeSH synonyms for better matching.

Authoritative definitions source
- NIH "RCDC Categories At a Glance" table data is served from:
  https://grants.nih.gov/json/public-def-categories.json

Notes
- NIH's official RCDC system uses an automated indexing tool and matches project text against
  concepts from the RCDC Thesaurus; this script is a best-effort approximation using the
  publicly available category summaries.

Dependencies
  pip install pandas requests scikit-learn

Usage examples
  python rcdc_classifier.py -i awards.csv -o awards_with_rcdc.csv
  python rcdc_classifier.py -i awards.tsv -o awards_with_rcdc.csv --top-k 5
  python rcdc_classifier.py -i awards.csv -o out.csv --threshold 0.03
  python rcdc_classifier.py -i awards.csv -o out.csv --no-rules --no-expansion
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RCDC_DEFINITIONS_URL = "https://grants.nih.gov/json/public-def-categories.json"

# Default: use the fields most likely to contain scientific content.
# Weights are a simple way to emphasize titles/abstracts without needing a custom model.
DEFAULT_FIELD_WEIGHTS: List[Tuple[str, int]] = [
    ("AWARD_LONG_TITLE", 3),
    ("PRPSL_LONG_TITLE", 3),
    ("PRPSL_ABSTRACT", 2),
    ("KEY_TERMS", 2),
    ("AWARD_CLASS_DES", 1),
    ("DIRECT_SPONSOR_DES", 1),
    ("SPONSOR_CLASS_FIN_DESCR", 1),
]

# Default threshold for multi-label output (categories with score >= threshold are included)
DEFAULT_MULTI_LABEL_THRESHOLD = 0.03

# Rule-based keyword-to-category mappings.
# If ANY keyword (case-insensitive) appears in award text, the category is guaranteed.
# Keywords should be specific enough to avoid false positives.
KEYWORD_RULES: Dict[str, List[str]] = {
    # Infectious diseases
    "HIV/AIDS": ["hiv", "aids", "human immunodeficiency virus", "antiretroviral"],
    "Malaria": ["malaria", "plasmodium", "antimalarial"],
    "Tuberculosis": ["tuberculosis", "mycobacterium tuberculosis", "tb infection", "latent tb"],
    "Hepatitis": ["hepatitis", "hcv", "hbv", "hepatitis c", "hepatitis b"],
    "Influenza": ["influenza", "flu virus", "h1n1", "h5n1", "avian flu"],
    "Emerging Infectious Diseases": ["emerging infection", "outbreak", "epidemic", "pandemic", "zoonotic"],
    "Infectious Diseases": ["pathogen", "bacterial infection", "viral infection", "infectious agent"],
    # Cancer types
    "Cancer": ["cancer", "tumor", "tumour", "oncology", "carcinoma", "malignancy", "neoplasm"],
    "Breast Cancer": ["breast cancer", "mammary tumor", "brca1", "brca2"],
    "Lung Cancer": ["lung cancer", "non-small cell lung", "small cell lung", "nsclc", "sclc"],
    "Prostate Cancer": ["prostate cancer", "prostatic neoplasm"],
    "Colorectal Cancer": ["colorectal cancer", "colon cancer", "rectal cancer"],
    "Pancreatic Cancer": ["pancreatic cancer", "pancreatic adenocarcinoma"],
    "Leukemia": ["leukemia", "leukaemia", "acute myeloid", "chronic lymphocytic"],
    "Lymphoma": ["lymphoma", "hodgkin", "non-hodgkin"],
    # Chronic diseases
    "Diabetes": ["diabetes", "diabetic", "insulin resistance", "hyperglycemia", "type 1 diabetes", "type 2 diabetes"],
    "Obesity": ["obesity", "obese", "adiposity", "body mass index", "bmi"],
    "Heart Disease": ["heart disease", "cardiac", "cardiovascular", "myocardial", "coronary artery"],
    "Stroke": ["stroke", "cerebrovascular", "ischemic stroke", "hemorrhagic stroke"],
    "Alzheimer's Disease": ["alzheimer", "alzheimer's", "amyloid", "tau protein", "dementia"],
    "Parkinson's Disease": ["parkinson", "parkinson's", "dopaminergic", "substantia nigra"],
    # Organ systems
    "Lung": ["lung", "pulmonary", "respiratory", "airway", "bronchial", "alveolar"],
    "Kidney Disease": ["kidney", "renal", "nephro", "glomerular", "dialysis"],
    "Liver Disease": ["liver", "hepatic", "cirrhosis", "fibrosis"],
    "Brain Disorders": ["brain", "cerebral", "neural", "neurological", "cns"],
    "Eye Disease and Disorders of Vision": ["eye", "ocular", "retina", "vision", "optic", "blindness"],
    # Research types
    "Clinical Research": ["clinical trial", "clinical study", "patient outcome", "randomized controlled"],
    "Clinical Trials and Supportive Activities": ["phase i", "phase ii", "phase iii", "phase 1", "phase 2", "phase 3"],
    "Genetics": ["genetic", "gene", "genomic", "dna", "mutation", "polymorphism", "gwas"],
    "Biotechnology": ["biotechnology", "biotech", "recombinant", "transgenic", "crispr", "gene editing",
                      "protein engineering", "enzyme engineering", "biocatalysis", "fermentation",
                      "cell culture", "bioproduction", "monoclonal antibody"],
    "Bioengineering": ["bioengineering", "biomedical engineering", "biomaterial", "tissue engineering",
                       "biofabrication", "scaffold", "hydrogel", "bioreactor", "microfluidic",
                       "lab-on-chip", "3d printing", "additive manufacturing", "medical device",
                       "implant", "prosthetic", "bioimaging", "biosensor", "crystallography",
                       "cryo-em", "cryoet", "x-ray diffraction", "synchrotron", "beamline"],
    "Neurosciences": ["neuroscience", "neuron", "synaptic", "neuronal", "brain circuit"],
    "Immunotherapy": ["immunotherapy", "car-t", "checkpoint inhibitor", "pd-1", "pd-l1", "ctla-4"],
    "Gene Therapy": ["gene therapy", "viral vector", "aav", "lentivirus", "gene transfer"],
    # Special populations
    "Pediatric": ["pediatric", "paediatric", "child", "children", "infant", "neonatal", "adolescent"],
    "Aging": ["aging", "ageing", "elderly", "geriatric", "older adult", "senescence"],
    "Women's Health Research": ["women's health", "female", "maternal", "pregnancy", "prenatal"],
    "Minority Health": ["health disparit", "minority health", "underserved", "health equity"],
    # Other important categories
    "Mental Health": ["mental health", "psychiatric", "depression", "anxiety", "schizophrenia", "bipolar"],
    "Substance Use": ["substance abuse", "addiction", "drug abuse", "alcohol abuse", "opioid"],
    "Pain Research": ["pain", "analges", "nocicepti", "chronic pain"],
    "Vaccine Related": ["vaccine", "vaccination", "immunization", "antigen", "adjuvant"],
    "Prevention": ["prevention", "preventive", "prophylaxis", "screening", "early detection"],
    # Technical/infrastructure categories
    "Nanotechnology": ["nanotechnology", "nanoparticle", "nanomaterial", "nanoscale",
                       "nanowire", "nanofiber", "nanomedicine", "nanocarrier"],
    "Networking and Information Technology R&D (NITRD)": ["data infrastructure", "cyberinfrastructure",
                                                          "high-performance computing", "hpc",
                                                          "data sharing", "data repository", "cloud computing",
                                                          "scientific computing", "computational pipeline"],
    "Data Science": ["data science", "big data", "data analytics", "machine learning pipeline",
                     "data integration", "data harmonization", "data curation"],
    # More specific disease/organ categories
    "Digestive Diseases": ["digestive", "gastrointestinal", "gi tract", "intestinal", "colon",
                           "stomach", "esophageal", "pancreatitis", "ibd", "crohn", "colitis"],
    "Hematology": ["hematology", "blood disorder", "hemostasis", "coagulation", "thrombosis",
                   "bleeding", "platelet", "hemophilia", "anemia", "sickle cell"],
    "Urologic Diseases": ["urologic", "bladder", "prostate", "urinary tract", "kidney stone"],
}

# MeSH-based synonym expansion dictionary.
# Maps common biomedical terms to their synonyms/related terms for query expansion.
MESH_SYNONYMS: Dict[str, List[str]] = {
    # Diseases
    "cancer": ["neoplasm", "tumor", "malignancy", "carcinoma", "oncology"],
    "tumor": ["neoplasm", "cancer", "malignancy", "mass", "growth"],
    "diabetes": ["diabetes mellitus", "hyperglycemia", "insulin resistance", "diabetic"],
    "hypertension": ["high blood pressure", "elevated blood pressure", "htn"],
    "obesity": ["overweight", "adiposity", "excess weight", "high bmi"],
    "infection": ["infectious disease", "pathogen", "microbial", "sepsis"],
    "inflammation": ["inflammatory", "inflammatory response", "inflamed"],
    "heart disease": ["cardiovascular disease", "cardiac disease", "coronary disease"],
    "stroke": ["cerebrovascular accident", "cva", "brain infarction", "ischemia"],
    "alzheimer": ["alzheimer's disease", "ad", "dementia", "cognitive decline"],
    "parkinson": ["parkinson's disease", "pd", "parkinsonian"],
    # Anatomy
    "brain": ["cerebral", "cerebrum", "cns", "central nervous system", "neural"],
    "heart": ["cardiac", "cardiovascular", "myocardial", "coronary"],
    "lung": ["pulmonary", "respiratory", "bronchial", "airway"],
    "liver": ["hepatic", "hepato"],
    "kidney": ["renal", "nephro"],
    "blood": ["hematologic", "hematopoietic", "vascular"],
    "bone": ["skeletal", "osseous", "osteo"],
    "muscle": ["muscular", "myogenic", "skeletal muscle"],
    "skin": ["dermal", "cutaneous", "epidermal"],
    "eye": ["ocular", "ophthalmic", "optic", "visual"],
    # Cell biology
    "cell": ["cellular", "cytoplasmic"],
    "protein": ["proteomic", "polypeptide", "enzyme"],
    "gene": ["genetic", "genomic", "dna"],
    "rna": ["transcript", "mrna", "non-coding rna"],
    "antibody": ["immunoglobulin", "ig", "monoclonal"],
    "receptor": ["ligand binding", "cell surface receptor"],
    "enzyme": ["catalytic", "enzymatic"],
    # Research methods
    "clinical trial": ["randomized controlled trial", "rct", "clinical study"],
    "mouse": ["murine", "mice", "rodent"],
    "rat": ["rodent", "murine model"],
    "in vitro": ["cell culture", "cultured cells"],
    "in vivo": ["animal model", "living organism"],
    # Treatments
    "therapy": ["treatment", "therapeutic", "intervention"],
    "drug": ["pharmaceutical", "medication", "compound", "agent"],
    "surgery": ["surgical", "operative", "resection"],
    "radiation": ["radiotherapy", "irradiation", "ionizing radiation"],
    "chemotherapy": ["cytotoxic", "antineoplastic", "anticancer drug"],
    "immunotherapy": ["immune therapy", "checkpoint inhibitor", "car-t"],
    "vaccine": ["vaccination", "immunization", "antigen"],
    # Populations
    "child": ["pediatric", "children", "infant", "juvenile"],
    "elderly": ["geriatric", "aged", "older adult", "senior"],
    "pregnant": ["pregnancy", "prenatal", "maternal", "gestational"],
    "women": ["female", "woman", "maternal"],
    "men": ["male", "man"],
}

_TAG_RE = re.compile(r"<[^>]+>")  # strip HTML tags from NIH definition summaries


def clean_text(value: object) -> str:
    """
    Normalize text:
      - convert to string
      - decode HTML entities
      - remove HTML tags
      - collapse whitespace
    """
    if value is None:
        return ""
    s = str(value)
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def expand_text_with_synonyms(text: str, synonyms: Dict[str, List[str]]) -> str:
    """
    Expand text by appending synonyms for recognized terms.
    This improves TF-IDF matching by ensuring related concepts share vocabulary.
    """
    if not text:
        return text

    text_lower = text.lower()
    expansions = [syn for term, syns in synonyms.items() if term in text_lower for syn in syns]

    return f"{text} {' '.join(expansions)}" if expansions else text


def apply_keyword_rules(
    text: str,
    rules: Dict[str, List[str]],
    valid_categories: set,
) -> List[str]:
    """
    Check text against keyword rules and return list of categories that should be assigned.
    Only returns categories that exist in the valid_categories set.
    """
    if not text:
        return []

    text_lower = text.lower()
    return [
        category
        for category, keywords in rules.items()
        if category in valid_categories and any(kw in text_lower for kw in keywords)
    ]


@dataclass(frozen=True)
class DefinitionsMeta:
    source_url: str
    retrieved_at_utc: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    note: Optional[str] = None


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_rcdc_definitions(cache_dir: Path, timeout_s: int = 30) -> Tuple[pd.DataFrame, DefinitionsMeta]:
    """
    Download the latest NIH RCDC category definitions (public category summaries).

    Behavior:
      - Uses a local cache file if present.
      - Revalidates on each run using If-None-Match / If-Modified-Since when possible.
      - If the download fails but cache exists, falls back to cached definitions.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_json = cache_dir / "public-def-categories.json"
    cache_meta = cache_dir / "public-def-categories.meta.json"

    headers: Dict[str, str] = {}
    if cache_meta.exists():
        try:
            meta_prev = _read_json(cache_meta)
            if isinstance(meta_prev, dict):
                if meta_prev.get("etag"):
                    headers["If-None-Match"] = meta_prev["etag"]
                if meta_prev.get("last_modified"):
                    headers["If-Modified-Since"] = meta_prev["last_modified"]
        except Exception:
            # Cache meta is optional; ignore if corrupted.
            pass

    try:
        resp = requests.get(RCDC_DEFINITIONS_URL, headers=headers, timeout=timeout_s)
        if resp.status_code == 304 and cache_json.exists():
            # Not modified; use cached file
            data = _read_json(cache_json)
            meta = _read_json(cache_meta) if cache_meta.exists() else {}
            meta_obj = DefinitionsMeta(
                source_url=RCDC_DEFINITIONS_URL,
                retrieved_at_utc=meta.get("retrieved_at_utc", "(cache)"),
                etag=meta.get("etag"),
                last_modified=meta.get("last_modified"),
                note="Used cached definitions (HTTP 304 Not Modified).",
            )
            return _definitions_df_from_payload(data), meta_obj

        resp.raise_for_status()
        data = resp.json()

        _write_json(cache_json, data)
        meta_obj = DefinitionsMeta(
            source_url=RCDC_DEFINITIONS_URL,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            etag=resp.headers.get("ETag"),
            last_modified=resp.headers.get("Last-Modified"),
        )
        _write_json(cache_meta, meta_obj.__dict__)
        return _definitions_df_from_payload(data), meta_obj

    except Exception as e:
        if cache_json.exists():
            data = _read_json(cache_json)
            meta_obj = DefinitionsMeta(
                source_url=RCDC_DEFINITIONS_URL,
                retrieved_at_utc="(cache)",
                note=f"Download failed; used cached definitions. Error: {type(e).__name__}: {e}",
            )
            return _definitions_df_from_payload(data), meta_obj
        raise RuntimeError(
            f"Failed to download RCDC definitions and no cache exists. "
            f"Error: {type(e).__name__}: {e}"
        ) from e


def _definitions_df_from_payload(payload: object) -> pd.DataFrame:
    """
    Convert the NIH definitions JSON payload into a DataFrame.
    Expected payload: list[ {name, created, category_summary, categories_included}, ... ]
    """
    if not isinstance(payload, list):
        raise ValueError("Unexpected RCDC definitions payload shape (expected a list).")

    df = pd.DataFrame(payload)
    for col in ("name", "created", "category_summary", "categories_included"):
        if col not in df.columns:
            raise ValueError(f"RCDC definitions missing expected column: {col!r}")

    df["category_summary_clean"] = df["category_summary"].apply(clean_text)
    df["categories_included_clean"] = df["categories_included"].apply(clean_text)

    # Definition document used for similarity scoring
    df["definition_text"] = (
        df["name"].map(clean_text)
        + " "
        + df["category_summary_clean"]
        + " Categories included: "
        + df["categories_included_clean"]
    )

    # Stable ordering
    df = df.sort_values(["name"]).reset_index(drop=True)
    return df


def read_awards_table(path: Path) -> pd.DataFrame:
    """
    Read awards file with automatic delimiter detection (handles comma-CSV and tab-TSV).
    All columns are read as strings to preserve IDs and avoid NaN surprises.
    """
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)


def build_award_texts(
    awards_df: pd.DataFrame,
    field_weights: Sequence[Tuple[str, int]],
    use_expansion: bool = True,
) -> List[str]:
    """
    Create one text document per award row using weighted concatenation of selected fields.
    Missing fields are ignored.

    Args:
        awards_df: DataFrame containing award data.
        field_weights: List of (field_name, weight) tuples.
        use_expansion: If True, expand text with MeSH synonyms for better matching.
    """

    fields_present = [f for f, _w in field_weights if f in awards_df.columns]

    if not fields_present:
        raise ValueError(
            "None of the requested text fields exist in the input file. "
            f"Requested: {[f for f, _ in field_weights]}"
        )

    weights_map = {f: w for f, w in field_weights if f in awards_df.columns and w > 0}

    def row_to_text(row: pd.Series) -> str:
        parts: List[str] = []
        for f, w in weights_map.items():
            val = clean_text(row.get(f, ""))
            # Skip short abstract values (likely placeholders like "None Provided")
            if f == "PRPSL_ABSTRACT" and len(val) < 30:
                continue
            if val:
                parts.extend([val] * w)
        text = " ".join(parts)

        # Apply MeSH synonym expansion
        if use_expansion and text:
            text = expand_text_with_synonyms(text, MESH_SYNONYMS)

        return text

    return awards_df.apply(row_to_text, axis=1).tolist()


def classify_awards(
    awards_df: pd.DataFrame,
    defs_df: pd.DataFrame,
    award_texts: List[str],
    top_k: int = 3,
    threshold: float = DEFAULT_MULTI_LABEL_THRESHOLD,
    use_rules: bool = True,
) -> pd.DataFrame:
    """
    Compute cosine similarity between each award text and each RCDC definition text.

    Returns awards_df with:
      - RCDC_CATEGORIES: semicolon-separated list of categories above threshold (multi-label)
      - RCDC_PRIMARY: highest-scoring category
      - RCDC_SCORE: score of primary category
      - RCDC_TOP_K: top-K categories with scores

    Args:
        awards_df: DataFrame containing award data.
        defs_df: DataFrame containing RCDC category definitions.
        award_texts: List of text documents (one per award).
        top_k: Number of top categories to include in RCDC_TOP_K column.
        threshold: Minimum similarity score for a category to be included in RCDC_CATEGORIES.
        use_rules: If True, also apply keyword-based rules to guarantee certain categories.
    """
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")

    category_texts = defs_df["definition_text"].tolist()
    category_names = defs_df["name"].tolist()
    n_cats = len(category_texts)
    n_awards = len(award_texts)

    valid_categories = set(category_names)
    category_index = {name: idx for idx, name in enumerate(category_names)}

    # Fit on combined corpus so vocabulary covers both definitions and awards.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        sublinear_tf=True,
    )

    combined = category_texts + award_texts
    X = vectorizer.fit_transform(combined)

    cat_matrix = X[:n_cats]
    award_matrix = X[n_cats:]

    sim = cosine_similarity(award_matrix, cat_matrix)  # (n_awards, n_cats)

    best_idx = sim.argmax(axis=1)
    best_score = sim[np.arange(n_awards), best_idx]

    out = awards_df.copy()

    # Build multi-label categories for each award
    multi_label_categories: List[str] = []
    for i in range(n_awards):
        award_text = award_texts[i]

        if not award_text.strip():
            # Empty text: no categories
            multi_label_categories.append("")
            continue

        # Get categories above threshold from TF-IDF similarity
        above_threshold_idx = np.where(sim[i] >= threshold)[0]
        tfidf_cats = {category_names[idx] for idx in above_threshold_idx}

        # Apply keyword rules to get additional guaranteed categories
        rule_cats = (
            set(apply_keyword_rules(award_text, KEYWORD_RULES, valid_categories))
            if use_rules
            else set()
        )

        # Combine TF-IDF and rule-based categories
        all_cats = tfidf_cats | rule_cats

        # Sort by similarity score (highest first), then alphabetically
        sorted_cats = sorted(all_cats, key=lambda cat: (-sim[i, category_index[cat]], cat))
        multi_label_categories.append("; ".join(sorted_cats))

    out["RCDC_CATEGORIES"] = multi_label_categories
    out["RCDC_PRIMARY"] = [category_names[i] for i in best_idx]
    out["RCDC_SCORE"] = best_score.astype(float)

    # Handle empty award texts: avoid bogus assignments
    empty_mask = np.array([not t.strip() for t in award_texts], dtype=bool)
    if empty_mask.any():
        out.loc[empty_mask, "RCDC_PRIMARY"] = ""
        out.loc[empty_mask, "RCDC_SCORE"] = 0.0

    # Top-K column for backward compatibility and debugging
    if top_k > 1:
        k = min(top_k, n_cats)
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_scores = np.take_along_axis(sim, topk_idx, axis=1)

        topk_str: List[str] = []
        for idxs, scores in zip(topk_idx, topk_scores):
            pairs = [f"{category_names[i]}:{scores[j]:.6f}" for j, i in enumerate(idxs)]
            topk_str.append("; ".join(pairs))

        out[f"RCDC_TOP_{k}"] = topk_str
        out.loc[empty_mask, f"RCDC_TOP_{k}"] = ""

    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify awards into NIH RCDC categories using current NIH definitions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rcdc_classifier.py -i awards.csv -o out.csv
  python rcdc_classifier.py -i awards.csv -o out.csv --threshold 0.05
  python rcdc_classifier.py -i awards.csv -o out.csv --no-rules --no-expansion
        """,
    )
    p.add_argument("-i", "--input", required=True, help="Path to awards CSV/TSV.")
    p.add_argument("-o", "--output", required=True, help="Path to output CSV.")
    p.add_argument("--cache-dir", default=".rcdc_cache", help="Directory for cached NIH definitions.")
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Also emit top-K categories with scores (default: 3). Use 1 to emit only the best category.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MULTI_LABEL_THRESHOLD,
        help=f"Minimum similarity score for multi-label output (default: {DEFAULT_MULTI_LABEL_THRESHOLD}).",
    )
    p.add_argument(
        "--no-rules",
        action="store_true",
        help="Disable rule-based keyword matching (only use TF-IDF similarity).",
    )
    p.add_argument(
        "--no-expansion",
        action="store_true",
        help="Disable MeSH synonym expansion for query text.",
    )
    p.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help=(
            "Optional override: list of award columns to use as text input (all weights=1). "
            "If omitted, a weighted default set is used."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    awards_df = read_awards_table(input_path)

    defs_df, defs_meta = fetch_rcdc_definitions(cache_dir=cache_dir)

    field_weights = [(f, 1) for f in args.fields] if args.fields else DEFAULT_FIELD_WEIGHTS
    use_expansion = not args.no_expansion
    use_rules = not args.no_rules

    # Build award texts with optional MeSH expansion
    award_texts = build_award_texts(
        awards_df,
        field_weights=field_weights,
        use_expansion=use_expansion,
    )

    out_df = classify_awards(
        awards_df,
        defs_df,
        award_texts,
        top_k=args.top_k,
        threshold=args.threshold,
        use_rules=use_rules,
    )

    # Write output
    out_df.to_csv(output_path, index=False)

    # Write run metadata next to output for reproducibility
    meta_out = output_path.with_suffix(output_path.suffix + ".rcdc_definitions_meta.json")
    _write_json(meta_out, defs_meta.__dict__)

    print(f"Wrote: {output_path}")
    print(f"Wrote: {meta_out}")
    print(f"Definitions source: {defs_meta.source_url}")
    print(f"Definitions retrieved_at_utc: {defs_meta.retrieved_at_utc}")
    print(f"Settings: threshold={args.threshold}, rules={'enabled' if use_rules else 'disabled'}, expansion={'enabled' if use_expansion else 'disabled'}")
    if defs_meta.etag:
        print(f"Definitions ETag: {defs_meta.etag}")
    if defs_meta.last_modified:
        print(f"Definitions Last-Modified: {defs_meta.last_modified}")
    if defs_meta.note:
        print(f"Note: {defs_meta.note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
