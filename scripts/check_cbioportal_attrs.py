import urllib.request
import json

url_attr = "https://www.cbioportal.org/api/studies/gbm_tcga/clinical-attributes"
req = urllib.request.Request(url_attr)
with urllib.request.urlopen(req) as resp:
    attrs = json.loads(resp.read().decode("utf-8"))

print(f"Total clinical attributes for gbm_tcga: {len(attrs)}")
for a in attrs:
    if any(k in a['displayName'].lower() or k in a['clinicalAttributeId'].lower() for k in ['idh', 'mgmt', 'kps', 'age', 'gender', 'survival']):
        print(f"ID: {a['clinicalAttributeId']:<30} | Name: {a['displayName']}")
