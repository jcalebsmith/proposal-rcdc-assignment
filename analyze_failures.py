#!/usr/bin/env python3
"""Analyze classifier performance against NIH gold standard with new multi-label features."""

import html
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Change to script directory
os.chdir(Path(__file__).parent)

_TAG_RE = re.compile(r"<[^>]+>")

DEFAULT_FIELD_WEIGHTS: List[Tuple[str, int]] = [
    ("AWARD_LONG_TITLE", 3),
    ("PRPSL_LONG_TITLE", 3),
    ("PRPSL_ABSTRACT", 2),
    ("KEY_TERMS", 2),
    ("AWARD_CLASS_DES", 1),
    ("DIRECT_SPONSOR_DES", 1),
    ("SPONSOR_CLASS_FIN_DESCR", 1),
]

DEFAULT_MULTI_LABEL_THRESHOLD = 0.03

# Import the keyword rules and MeSH synonyms from the main module
KEYWORD_RULES: Dict[str, List[str]] = {
    "HIV/AIDS": ["hiv", "aids", "human immunodeficiency virus", "antiretroviral"],
    "Malaria": ["malaria", "plasmodium", "antimalarial"],
    "Tuberculosis": ["tuberculosis", "mycobacterium tuberculosis", "tb infection", "latent tb"],
    "Hepatitis": ["hepatitis", "hcv", "hbv", "hepatitis c", "hepatitis b"],
    "Influenza": ["influenza", "flu virus", "h1n1", "h5n1", "avian flu"],
    "Emerging Infectious Diseases": ["emerging infection", "outbreak", "epidemic", "pandemic", "zoonotic"],
    "Infectious Diseases": ["pathogen", "bacterial infection", "viral infection", "infectious agent"],
    "Cancer": ["cancer", "tumor", "tumour", "oncology", "carcinoma", "malignancy", "neoplasm"],
    "Breast Cancer": ["breast cancer", "mammary tumor", "brca1", "brca2"],
    "Lung Cancer": ["lung cancer", "non-small cell lung", "small cell lung", "nsclc", "sclc"],
    "Prostate Cancer": ["prostate cancer", "prostatic neoplasm"],
    "Colorectal Cancer": ["colorectal cancer", "colon cancer", "rectal cancer"],
    "Pancreatic Cancer": ["pancreatic cancer", "pancreatic adenocarcinoma"],
    "Leukemia": ["leukemia", "leukaemia", "acute myeloid", "chronic lymphocytic"],
    "Lymphoma": ["lymphoma", "hodgkin", "non-hodgkin"],
    "Diabetes": ["diabetes", "diabetic", "insulin resistance", "hyperglycemia", "type 1 diabetes", "type 2 diabetes"],
    "Obesity": ["obesity", "obese", "adiposity", "body mass index", "bmi"],
    "Heart Disease": ["heart disease", "cardiac", "cardiovascular", "myocardial", "coronary artery"],
    "Stroke": ["stroke", "cerebrovascular", "ischemic stroke", "hemorrhagic stroke"],
    "Alzheimer's Disease": ["alzheimer", "alzheimer's", "amyloid", "tau protein", "dementia"],
    "Parkinson's Disease": ["parkinson", "parkinson's", "dopaminergic", "substantia nigra"],
    "Lung": ["lung", "pulmonary", "respiratory", "airway", "bronchial", "alveolar"],
    "Kidney Disease": ["kidney", "renal", "nephro", "glomerular", "dialysis"],
    "Liver Disease": ["liver", "hepatic", "cirrhosis", "fibrosis"],
    "Brain Disorders": ["brain", "cerebral", "neural", "neurological", "cns"],
    "Eye Disease and Disorders of Vision": ["eye", "ocular", "retina", "vision", "optic", "blindness"],
    "Clinical Research": ["clinical trial", "clinical study", "patient outcome", "randomized controlled"],
    "Clinical Trials and Supportive Activities": ["phase i", "phase ii", "phase iii", "phase 1", "phase 2", "phase 3"],
    "Genetics": ["genetic", "gene", "genomic", "dna", "mutation", "polymorphism", "gwas"],
    "Biotechnology": ["biotechnology", "biotech", "recombinant", "transgenic", "crispr", "gene editing"],
    "Bioengineering": ["bioengineering", "biomedical engineering", "biomaterial", "tissue engineering"],
    "Neurosciences": ["neuroscience", "neuron", "synaptic", "neuronal", "brain circuit"],
    "Immunotherapy": ["immunotherapy", "car-t", "checkpoint inhibitor", "pd-1", "pd-l1", "ctla-4"],
    "Gene Therapy": ["gene therapy", "viral vector", "aav", "lentivirus", "gene transfer"],
    "Pediatric": ["pediatric", "paediatric", "child", "children", "infant", "neonatal", "adolescent"],
    "Aging": ["aging", "ageing", "elderly", "geriatric", "older adult", "senescence"],
    "Women's Health Research": ["women's health", "female", "maternal", "pregnancy", "prenatal"],
    "Minority Health": ["health disparit", "minority health", "underserved", "health equity"],
    "Mental Health": ["mental health", "psychiatric", "depression", "anxiety", "schizophrenia", "bipolar"],
    "Substance Use": ["substance abuse", "addiction", "drug abuse", "alcohol abuse", "opioid"],
    "Pain Research": ["pain", "analges", "nocicepti", "chronic pain"],
    "Vaccine Related": ["vaccine", "vaccination", "immunization", "antigen", "adjuvant"],
    "Prevention": ["prevention", "preventive", "prophylaxis", "screening", "early detection"],
    # Technical/infrastructure categories (often missed)
    "Bioengineering": ["bioengineering", "biomaterial", "tissue engineering", "biofabrication",
                       "scaffold", "hydrogel", "bioreactor", "microfluidic", "lab-on-chip",
                       "3d printing", "additive manufacturing", "medical device", "implant",
                       "prosthetic", "bioimaging", "biosensor", "crystallography", "cryo-em",
                       "cryoet", "x-ray diffraction", "synchrotron", "beamline"],
    "Biotechnology": ["biotechnology", "biotech", "recombinant", "transgenic", "crispr",
                      "gene editing", "protein engineering", "enzyme engineering", "biocatalysis",
                      "fermentation", "cell culture", "bioproduction", "monoclonal antibody"],
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

MESH_SYNONYMS: Dict[str, List[str]] = {
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
    "cell": ["cellular", "cytoplasmic"],
    "protein": ["proteomic", "polypeptide", "enzyme"],
    "gene": ["genetic", "genomic", "dna"],
    "rna": ["transcript", "mrna", "non-coding rna"],
    "antibody": ["immunoglobulin", "ig", "monoclonal"],
    "receptor": ["ligand binding", "cell surface receptor"],
    "enzyme": ["catalytic", "enzymatic"],
    "clinical trial": ["randomized controlled trial", "rct", "clinical study"],
    "mouse": ["murine", "mice", "rodent"],
    "rat": ["rodent", "murine model"],
    "in vitro": ["cell culture", "cultured cells"],
    "in vivo": ["animal model", "living organism"],
    "therapy": ["treatment", "therapeutic", "intervention"],
    "drug": ["pharmaceutical", "medication", "compound", "agent"],
    "surgery": ["surgical", "operative", "resection"],
    "radiation": ["radiotherapy", "irradiation", "ionizing radiation"],
    "chemotherapy": ["cytotoxic", "antineoplastic", "anticancer drug"],
    "immunotherapy": ["immune therapy", "checkpoint inhibitor", "car-t"],
    "vaccine": ["vaccination", "immunization", "antigen"],
    "child": ["pediatric", "children", "infant", "juvenile"],
    "elderly": ["geriatric", "aged", "older adult", "senior"],
    "pregnant": ["pregnancy", "prenatal", "maternal", "gestational"],
    "women": ["female", "woman", "maternal"],
    "men": ["male", "man"],
}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    s = str(value)
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def expand_text_with_synonyms(text: str, synonyms: Dict[str, List[str]]) -> str:
    if not text:
        return text
    text_lower = text.lower()
    expansions: List[str] = []
    for term, syns in synonyms.items():
        if term in text_lower:
            expansions.extend(syns)
    if expansions:
        return text + " " + " ".join(expansions)
    return text


def apply_keyword_rules(text: str, rules: Dict[str, List[str]], valid_categories: set) -> List[str]:
    if not text:
        return []
    text_lower = text.lower()
    matched_categories: List[str] = []
    for category, keywords in rules.items():
        if category not in valid_categories:
            continue
        for keyword in keywords:
            if keyword in text_lower:
                matched_categories.append(category)
                break
    return matched_categories


def build_award_texts(
    awards_df: pd.DataFrame,
    field_weights: Sequence[Tuple[str, int]],
    use_expansion: bool = True,
) -> List[str]:
    fields_present = [f for f, _w in field_weights if f in awards_df.columns]
    if not fields_present:
        raise ValueError("None of the requested text fields exist in the input file.")
    weights_map = {f: w for f, w in field_weights if f in awards_df.columns and w > 0}

    def row_to_text(row: pd.Series) -> str:
        parts: List[str] = []
        for f, w in weights_map.items():
            val = clean_text(row.get(f, ""))
            if f == "PRPSL_ABSTRACT" and len(val) < 30:
                continue
            if val:
                parts.extend([val] * w)
        text = " ".join(parts)
        if use_expansion and text:
            text = expand_text_with_synonyms(text, MESH_SYNONYMS)
        return text

    return awards_df.apply(row_to_text, axis=1).tolist()


def classify_awards(
    awards_df: pd.DataFrame,
    defs_df: pd.DataFrame,
    award_texts: List[str],
    threshold: float = DEFAULT_MULTI_LABEL_THRESHOLD,
    use_rules: bool = True,
) -> pd.DataFrame:
    category_texts = defs_df["definition_text"].tolist()
    category_names = defs_df["name"].tolist()
    n_cats = len(category_texts)
    n_awards = len(award_texts)
    valid_categories = set(category_names)

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

    sim = cosine_similarity(award_matrix, cat_matrix)

    best_idx = sim.argmax(axis=1)
    best_score = sim[np.arange(n_awards), best_idx]

    out = awards_df.copy()

    multi_label_categories: List[str] = []
    for i in range(n_awards):
        award_text = award_texts[i]

        if not award_text.strip():
            multi_label_categories.append("")
            continue

        above_threshold_idx = np.where(sim[i] >= threshold)[0]
        tfidf_cats = {category_names[idx] for idx in above_threshold_idx}

        if use_rules:
            rule_cats = set(apply_keyword_rules(award_text, KEYWORD_RULES, valid_categories))
        else:
            rule_cats = set()

        all_cats = tfidf_cats | rule_cats

        def sort_key(cat: str) -> Tuple[float, str]:
            try:
                idx = category_names.index(cat)
                return (-sim[i, idx], cat)
            except ValueError:
                return (0.0, cat)

        sorted_cats = sorted(all_cats, key=sort_key)
        multi_label_categories.append("; ".join(sorted_cats))

    out["RCDC_CATEGORIES"] = multi_label_categories
    out["RCDC_PRIMARY"] = [category_names[i] for i in best_idx]
    out["RCDC_SCORE"] = best_score.astype(float)

    return out


def evaluate_accuracy(results_df: pd.DataFrame, config_name: str) -> dict:
    """Evaluate accuracy against gold standard."""
    correct_primary = 0
    correct_multilabel = 0
    total = 0
    failures = []

    for idx, row in results_df.iterrows():
        gold_cats_str = row.get("GOLD_STANDARD_CATEGORIES", "")
        if not gold_cats_str or pd.isna(gold_cats_str):
            continue

        total += 1
        gold_set = {c.strip() for c in gold_cats_str.split(";")}

        # Check primary (top-1) accuracy
        primary = row.get("RCDC_PRIMARY", "")
        if primary in gold_set:
            correct_primary += 1

        # Check multi-label accuracy (any overlap)
        predicted_str = row.get("RCDC_CATEGORIES", "")
        predicted_set = {c.strip() for c in predicted_str.split(";") if c.strip()}

        overlap = gold_set & predicted_set
        if overlap:
            correct_multilabel += 1
        else:
            failures.append({
                "idx": idx,
                "title": row.get("AWARD_LONG_TITLE", row.get("PRPSL_LONG_TITLE", ""))[:60],
                "primary": primary,
                "predicted": list(predicted_set)[:5],
                "gold": list(gold_set)[:5],
            })

    return {
        "config": config_name,
        "total": total,
        "primary_correct": correct_primary,
        "primary_accuracy": 100 * correct_primary / total if total > 0 else 0,
        "multilabel_correct": correct_multilabel,
        "multilabel_accuracy": 100 * correct_multilabel / total if total > 0 else 0,
        "failures": failures,
    }


# Load definitions from cache
cache_json = Path(".rcdc_cache/public-def-categories.json")
with open(cache_json, "r", encoding="utf-8") as f:
    defs_data = json.load(f)

defs_df = pd.DataFrame(defs_data)
defs_df["category_summary_clean"] = defs_df["category_summary"].apply(clean_text)
defs_df["categories_included_clean"] = defs_df["categories_included"].apply(clean_text)
defs_df["definition_text"] = (
    defs_df["name"].map(clean_text)
    + " "
    + defs_df["category_summary_clean"]
    + " Categories included: "
    + defs_df["categories_included_clean"]
)
defs_df = defs_df.sort_values(["name"]).reset_index(drop=True)

# Load gold standard - use expanded dataset if available
expanded_path = Path("data/nih_gold_standard_expanded.csv")
original_path = Path("data/nih_gold_standard.csv")

if expanded_path.exists():
    gold_df = pd.read_csv(expanded_path, dtype=str, keep_default_na=False)
    print(f"Gold standard (expanded): {len(gold_df)} projects")
else:
    gold_df = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    print(f"Gold standard (original): {len(gold_df)} projects")
print(f"RCDC categories: {len(defs_df)}")

# Test different configurations
configs = [
    {"name": "Baseline (TF-IDF only)", "use_expansion": False, "use_rules": False, "threshold": 0.03},
    {"name": "TF-IDF + Rules", "use_expansion": False, "use_rules": True, "threshold": 0.03},
    {"name": "TF-IDF + Expansion", "use_expansion": True, "use_rules": False, "threshold": 0.03},
    {"name": "Full (TF-IDF + Rules + Expansion)", "use_expansion": True, "use_rules": True, "threshold": 0.03},
    {"name": "Full with threshold=0.02", "use_expansion": True, "use_rules": True, "threshold": 0.02},
    {"name": "Full with threshold=0.05", "use_expansion": True, "use_rules": True, "threshold": 0.05},
]

print("\n" + "="*80)
print("TESTING CONFIGURATIONS")
print("="*80)

results_summary = []

for config in configs:
    print(f"\nTesting: {config['name']}")

    award_texts = build_award_texts(
        gold_df,
        field_weights=DEFAULT_FIELD_WEIGHTS,
        use_expansion=config["use_expansion"],
    )

    results_df = classify_awards(
        gold_df,
        defs_df,
        award_texts,
        threshold=config["threshold"],
        use_rules=config["use_rules"],
    )

    eval_result = evaluate_accuracy(results_df, config["name"])
    results_summary.append(eval_result)

    print(f"  Primary (Top-1) Accuracy: {eval_result['primary_accuracy']:.1f}% ({eval_result['primary_correct']}/{eval_result['total']})")
    print(f"  Multi-label Accuracy:     {eval_result['multilabel_accuracy']:.1f}% ({eval_result['multilabel_correct']}/{eval_result['total']})")

# Summary table
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Configuration':<45} {'Primary':>10} {'Multi-label':>12}")
print("-"*80)
for r in results_summary:
    print(f"{r['config']:<45} {r['primary_accuracy']:>9.1f}% {r['multilabel_accuracy']:>11.1f}%")

# Show failures for best configuration
best_config = max(results_summary, key=lambda x: x["multilabel_accuracy"])
print(f"\n" + "="*80)
print(f"FAILURES FOR BEST CONFIG: {best_config['config']}")
print("="*80)

for f in best_config["failures"]:
    print(f"\n--- #{f['idx']} ---")
    print(f"Title: {f['title']}...")
    print(f"Primary predicted: {f['primary']}")
    print(f"Predicted categories: {', '.join(f['predicted'])}")
    print(f"Gold categories: {', '.join(f['gold'])}")

# Save results
with open("data/accuracy_results.json", "w") as f_out:
    # Remove failures from summary for clean JSON
    clean_summary = [{k: v for k, v in r.items() if k != "failures"} for r in results_summary]
    json.dump(clean_summary, f_out, indent=2)

print(f"\nResults saved to data/accuracy_results.json")
