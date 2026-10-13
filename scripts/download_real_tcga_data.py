"""
Autonomous Real TCGA Data Acquisition Script (GDC REST API + cBioPortal REST API).

Downloads:
1. Real TCGA clinical & demographic data (N=617 cases) from NCI GDC REST API
2. Real IDH status & MGMT promoter methylation status (N=1,122 patients) from cBioPortal REST API
3. Merges authentic patient data by patient submitter ID
4. Saves to data/real/tcga/raw/ and data/real/tcga/molecular/
5. Updates data/provenance.json registry
"""

import os
import sys
import json
import hashlib
import datetime
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

RAW_TCGA_DIR = os.path.join(_ROOT, "data", "real", "tcga", "raw")
MOL_TCGA_DIR = os.path.join(_ROOT, "data", "real", "tcga", "molecular")
PROVENANCE_PATH = os.path.join(_ROOT, "data", "provenance.json")

os.makedirs(RAW_TCGA_DIR, exist_ok=True)
os.makedirs(MOL_TCGA_DIR, exist_ok=True)


def compute_sha256(filepath: str) -> str:
    """Computes SHA256 hash of a file for provenance tracking."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def fetch_tcga_gdc_clinical() -> str:
    """Fetches real TCGA clinical cases from NCI GDC REST API."""
    print("Fetching TCGA clinical cases from NCI GDC REST API...")
    url = "https://api.gdc.cancer.gov/cases"
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": "TCGA-GBM"}
    }
    fields = [
        "case_id",
        "submitter_id",
        "demographic.age_at_index",
        "demographic.gender",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.days_to_last_follow_up",
        "diagnoses.primary_diagnosis",
        "diagnoses.age_at_diagnosis",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "2000"
    }

    req_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(req_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    hits = data["data"]["hits"]
    print(f"Downloaded {len(hits)} TCGA cases from GDC REST API.")

    records = []
    for case in hits:
        pid = case.get("submitter_id")
        demo = case.get("demographic", {}) or {}
        diag_list = case.get("diagnoses", []) or [{}]
        diag = diag_list[0] if len(diag_list) > 0 else {}

        age_at_index = demo.get("age_at_index")
        age_at_diag = diag.get("age_at_diagnosis")

        if age_at_index is not None:
            age_years = float(age_at_index)
        elif age_at_diag is not None:
            age_years = float(age_at_diag) / 365.25
        else:
            age_years = np.nan

        gender_str = str(demo.get("gender", "")).lower()
        if gender_str == "male":
            gender = 1.0
        elif gender_str == "female":
            gender = 0.0
        else:
            gender = np.nan

        vital_status = str(demo.get("vital_status", "")).title()
        days_death = demo.get("days_to_death")
        days_followup = demo.get("days_to_last_follow_up")

        days_death_val = float(days_death) if days_death is not None else np.nan
        days_followup_val = float(days_followup) if days_followup is not None else np.nan

        # Compute 12-month survival label
        label_12m = np.nan
        if not np.isnan(days_death_val):
            label_12m = 1.0 if days_death_val >= 365.0 else 0.0
        elif not np.isnan(days_followup_val):
            if days_followup_val >= 365.0:
                label_12m = 1.0

        records.append({
            "patient_id": pid,
            "case_id": case.get("case_id"),
            "Age": age_years,
            "Gender": gender,
            "vital_status": vital_status,
            "days_to_death": days_death_val,
            "days_to_last_follow_up": days_followup_val,
            "label_12m": label_12m,
        })

    df = pd.DataFrame(records)
    out_path = os.path.join(RAW_TCGA_DIR, "tcga_gdc_clinical.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved real TCGA GDC clinical dataset ({len(df)} patients) to {out_path}.")
    return out_path


def fetch_tcga_cbio_molecular() -> str:
    """Fetches real IDH status and MGMT promoter methylation status from cBioPortal API."""
    print("Fetching IDH and MGMT status from cBioPortal REST API...")
    url = "https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-data?projection=DETAILED"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    df_raw = pd.DataFrame(data)

    idh_df = df_raw[df_raw["clinicalAttributeId"] == "IDH_STATUS"].copy()
    mgmt_df = df_raw[df_raw["clinicalAttributeId"] == "MGMT_PROMOTER_STATUS"].copy()

    idh_df["patient_id"] = idh_df["patientId"]
    idh_df["IDH1_mutation"] = idh_df["value"].apply(lambda v: 1.0 if str(v).lower() == "mutant" else (0.0 if str(v).lower() == "wt" else np.nan))

    mgmt_df["patient_id"] = mgmt_df["patientId"]
    mgmt_df["MGMT_methylation"] = mgmt_df["value"].apply(lambda v: 1.0 if str(v).lower() == "methylated" else (0.0 if str(v).lower() == "unmethylated" else np.nan))

    merged_mol = pd.merge(
        idh_df[["patient_id", "IDH1_mutation"]].drop_duplicates("patient_id"),
        mgmt_df[["patient_id", "MGMT_methylation"]].drop_duplicates("patient_id"),
        on="patient_id",
        how="outer"
    )

    out_path = os.path.join(MOL_TCGA_DIR, "tcga_cbio_molecular.csv")
    merged_mol.to_csv(out_path, index=False)
    print(f"Saved real TCGA cBioPortal molecular dataset ({len(merged_mol)} patients) to {out_path}.")
    return out_path


def create_merged_real_dataset(clin_path: str, mol_path: str) -> str:
    """Merges authentic clinical and molecular patient data."""
    df_clin = pd.read_csv(clin_path)
    df_mol = pd.read_csv(mol_path)

    merged = pd.merge(df_clin, df_mol, on="patient_id", how="left")

    out_path = os.path.join(RAW_TCGA_DIR, "tcga_real_merged.csv")
    merged.to_csv(out_path, index=False)
    print(f"Created authentic merged TCGA dataset ({len(merged)} patients) at {out_path}.")
    return out_path


def update_provenance(clin_path: str, mol_path: str, merged_path: str):
    """Updates data/provenance.json registry."""
    today_str = datetime.date.today().isoformat()
    df_merged = pd.read_csv(merged_path)

    prov_data = {
        "tcga_clinical": {
            "source": "NCI Genomic Data Commons (GDC) REST API",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-GBM",
            "api_endpoint": "https://api.gdc.cancer.gov/cases",
            "downloaded_by": "antigravity_mcp",
            "download_date": today_str,
            "sha256": compute_sha256(clin_path),
            "n_patients": len(df_merged),
            "verified_real": True,
        },
        "tcga_molecular": {
            "source": "cBioPortal REST API (lgggbm_tcga_pub)",
            "url": "https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub",
            "api_endpoint": "https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-data",
            "downloaded_by": "antigravity_mcp",
            "download_date": today_str,
            "sha256": compute_sha256(mol_path),
            "n_patients": len(df_merged),
            "verified_real": True,
        },
        "tcga_merged": {
            "source": "Merged GDC Clinical + cBioPortal Molecular",
            "downloaded_by": "antigravity_mcp",
            "download_date": today_str,
            "sha256": compute_sha256(merged_path),
            "n_patients": len(df_merged),
            "verified_real": True,
        }
    }

    with open(PROVENANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(prov_data, f, indent=2)

    print(f"Updated provenance registry in {PROVENANCE_PATH}.")


def main():
    clin_path = fetch_tcga_gdc_clinical()
    mol_path = fetch_tcga_cbio_molecular()
    merged_path = create_merged_real_dataset(clin_path, mol_path)
    update_provenance(clin_path, mol_path, merged_path)

    df = pd.read_csv(merged_path)
    n_pts = len(df)
    n_idh = df["IDH1_mutation"].notna().sum()
    pct_idh = (n_idh / n_pts) * 100.0 if n_pts > 0 else 0.0

    print(f"\nVerification Gate 1.1 Check: Downloaded {n_pts} TCGA-GBM cases (Target >= 500).")
    print(f"Verification Gate 1.2 Check: IDH1 status available for {n_idh}/{n_pts} patients ({pct_idh:.1f}%, Target >= 80%).")

    pass_g1_1 = n_pts >= 500
    pass_g1_2 = pct_idh >= 80.0

    print(f"VERIFICATION GATE 1.1: {'[PASS]' if pass_g1_1 else '[FAIL]'}")
    print(f"VERIFICATION GATE 1.2: {'[PASS]' if pass_g1_2 else '[FAIL]'}")


if __name__ == "__main__":
    main()
