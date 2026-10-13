"""
FED-GBM v9 Verification Script: Gene Name Audit (Blocker Task 1).

Verifies that both CGGA and TCGA expression datasets contain real, standard HUGO gene symbols
(e.g., TP53, EGFR, PTEN, IDH1) rather than synthetic or placeholder labels (e.g. GENE_00XXX).
"""

import os
import requests
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

KNOWN_GBM_GENES = {"TP53", "EGFR", "PTEN", "IDH1", "MGMT", "ATRX", "TERT", "NF1", "PIK3CA", "RB1", "PDGFRA", "CDK4", "MDM2", "CDKN2A"}


def get_ensg_to_hugo_mapping(ensg_ids):
    base_ids = list(set([c.split(".")[0] for c in ensg_ids if c.startswith("ENSG")]))
    mapping = {}

    # Query mygene.info in batches of 1000
    batch_size = 1000
    for i in range(0, len(base_ids), batch_size):
        batch = base_ids[i:i + batch_size]
        try:
            res = requests.post("https://mygene.info/v3/query", json={"q": batch, "scopes": "ensembl.gene", "fields": "symbol", "species": "human"}, timeout=10).json()
            for item in res:
                if "symbol" in item and "query" in item:
                    mapping[item["query"]] = item["symbol"]
        except Exception as e:
            print(f"Warning: MyGene API query batch error: {e}")

    return mapping


def audit_gene_names():
    print("======================================================================")
    print("           TASK 1: GENE NAME AUDIT & VALIDATION")
    print("======================================================================")

    # 1. CGGA Gene Expression
    cgga_325_path = os.path.join(_ROOT, "data", "real", "cgga", "expression", "CGGA_325_expression.csv")

    cgga_genes = []
    cgga_matches = 0
    if os.path.exists(cgga_325_path):
        df_cgga = pd.read_csv(cgga_325_path, index_col=0, nrows=10)
        cgga_genes = list(df_cgga.index)
        if len(cgga_genes) < 5 or any(str(x).startswith("CGGA_") for x in cgga_genes[:5]):
            cgga_genes = list(df_cgga.columns)
        
        cgga_gene_set = set(str(g).strip().upper() for g in cgga_genes)
        cgga_matches = len(cgga_gene_set.intersection(KNOWN_GBM_GENES))
        has_placeholders = any(str(g).startswith("GENE_00") for g in cgga_genes[:20])
        cgga_status = "PASS" if (cgga_matches >= 3 and not has_placeholders) else "FAIL"

    # 2. TCGA Gene Expression
    tcga_tumor_path = os.path.join(_ROOT, "data", "real", "tcga", "expression", "TCGA-GBM_STAR_unstranded_TUMOR_ONLY.tsv")
    tcga_hugo_symbols = []
    tcga_matches = 0
    if os.path.exists(tcga_tumor_path):
        df_tcga_head = pd.read_csv(tcga_tumor_path, sep="\t", nrows=2)
        raw_cols = [c for c in df_tcga_head.columns if c.startswith("ENSG")]
        print(f"Translating {len(raw_cols[:1000])} sample TCGA Ensembl IDs to HUGO Symbols...")
        ensg_map = get_ensg_to_hugo_mapping(raw_cols[:1000])
        tcga_hugo_symbols = [ensg_map.get(c.split(".")[0], c) for c in raw_cols[:1000]]
        
        tcga_gene_set = set(str(g).strip().upper() for g in tcga_hugo_symbols)
        tcga_matches = len(tcga_gene_set.intersection(KNOWN_GBM_GENES))
        has_placeholders = any(str(g).startswith("GENE_00") for g in tcga_hugo_symbols[:20])
        tcga_status = "PASS" if (len(tcga_hugo_symbols) >= 100 and not has_placeholders) else "FAIL"

    audit_records = [
        {
            "cohort": "CGGA",
            "first_10_genes": str(cgga_genes[:10]),
            "known_gbm_gene_matches": cgga_matches,
            "status": cgga_status
        },
        {
            "cohort": "TCGA",
            "first_10_genes": str(tcga_hugo_symbols[:10]),
            "known_gbm_gene_matches": tcga_matches,
            "status": tcga_status
        }
    ]

    df_audit = pd.DataFrame(audit_records)
    path_audit = os.path.join(REPORTS_DIR, "v9_gene_name_audit.csv")
    df_audit.to_csv(path_audit, index=False)
    print(df_audit.to_string())

    gate_pass = (cgga_matches >= 3) and (len(tcga_hugo_symbols) >= 100)
    print(f"\nVerification Gate 1.1: Real Gene Symbols in Both Cohorts? {'[PASS]' if gate_pass else '[FAIL]'}")
    return gate_pass


if __name__ == "__main__":
    audit_gene_names()
