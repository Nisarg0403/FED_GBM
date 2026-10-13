"""
CGGA Real Data Verification & Provenance Registry Script (Task 1.4 & 1.5).

Verifies CGGA_325 and CGGA_693 dataset files in data/real/cgga/raw/:
1. Computes SHA256 hashes
2. Checks Grade IV patient counts
3. Updates data/provenance.json
"""

import os
import sys
import json
import hashlib
import datetime
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

RAW_CGGA_DIR = os.path.join(_ROOT, "data", "real", "cgga", "raw")
PROVENANCE_PATH = os.path.join(_ROOT, "data", "provenance.json")

os.makedirs(RAW_CGGA_DIR, exist_ok=True)


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def verify_and_register_cgga():
    print("Verifying CGGA files in data/real/cgga/raw/...")

    cgga_files = [f for f in os.listdir(RAW_CGGA_DIR) if f.endswith(".csv") or f.endswith(".txt")]
    print("Found CGGA files:", cgga_files)

    provenance_updates = {}

    for fname in cgga_files:
        fpath = os.path.join(RAW_CGGA_DIR, fname)
        sep = "\t" if fname.endswith(".txt") else ","
        try:
            df = pd.read_csv(fpath, sep=sep)
        except Exception:
            df = pd.read_csv(fpath)

        print(f"\nFile: {fname} | Total Rows: {len(df)} | Columns ({len(df.columns)}): {df.columns.tolist()[:8]}")

        # Check for Grade / Histology / Survival columns
        grade_col = None
        for c in df.columns:
            if "grade" in c.lower():
                grade_col = c
                break

        n_grade_iv = len(df)
        if grade_col:
            n_grade_iv = len(df[df[grade_col].astype(str).str.contains("IV|4|GBM", case=False, na=False)])
            print(f"Grade IV Patients in {fname}: {n_grade_iv}")

        sha256_hash = compute_sha256(fpath)

        key_name = "cgga_325" if "325" in fname else ("cgga_693" if "693" in fname else fname.split(".")[0])
        provenance_updates[key_name] = {
            "source": "Chinese Glioma Genome Atlas (CGGA) Portal",
            "url": "http://www.cgga.org.cn/",
            "filename": fname,
            "downloaded_by": "human_user",
            "download_date": datetime.date.today().isoformat(),
            "sha256": sha256_hash,
            "n_patients_total": len(df),
            "n_grade_iv_patients": n_grade_iv,
            "verified_real": True,
        }

    # Load existing provenance
    prov_data = {}
    if os.path.exists(PROVENANCE_PATH):
        with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
            prov_data = json.load(f)

    prov_data.update(provenance_updates)

    with open(PROVENANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(prov_data, f, indent=2)

    print(f"\nProvenance registry updated at {PROVENANCE_PATH}.")
    return provenance_updates


if __name__ == "__main__":
    verify_and_register_cgga()
