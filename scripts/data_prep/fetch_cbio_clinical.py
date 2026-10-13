import urllib.request
import json
import pandas as pd

url_data = "https://www.cbioportal.org/api/studies/gbm_tcga/clinical-data/fetch"
body = json.dumps({"attributeIds": []}).encode("utf-8") # Empty list fetches all attributes
req = urllib.request.Request(url_data, data=body, headers={"Content-Type": "application/json"})

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print(f"Total clinical data points fetched: {len(data)}")

df = pd.DataFrame(data)
print("Unique attributes:", df["clinicalAttributeId"].unique().tolist())

# Pivot to sample x attribute table
pivot_df = df.pivot(index="sampleId", columns="clinicalAttributeId", values="value").reset_index()
print("\nPivoted shape:", pivot_df.shape)
print("\nFirst 5 rows:")
print(pivot_df.head())
