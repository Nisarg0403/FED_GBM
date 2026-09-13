# UPenn MRI Pipeline Validation Report (`UPENN_MRI_PIPELINE_VALIDATION.md`)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v3`)  
**Scope**: Technical validation of spatial orientation, voxel resampling, bias field correction, intensity normalization, and PyRadiomics extraction on representative UPenn MRI volumes.  
**Validation Date**: September 2, 2026  

---

## 1. Pilot Sample Selection & Orientation Audit

A representative subset of 3 UPenn NIfTI cases (`UPenn-GBM-00001`, `UPenn-GBM-00002`, `UPenn-GBM-00005`) was selected to validate the V3 preprocessing and extraction pipeline before cohort-wide processing.

### Image Header Properties:
- **Spatial Orientation**: Originally in `RAI` / `LAS` coordinate frames; standardized to canonical **`RAS` (Right-Anterior-Superior)** orientation using `SimpleITK.DICOMOrient`.
- **Original Voxel Spacing**: $0.9375 \times 0.9375 \times 1.0\,\text{mm}^3$ (T1, T1ce, T2, FLAIR).
- **Matrix Dimensions**: $240 \times 240 \times 155$ voxels.
- **Data Types**: `Float32` / `Int16`.

---

## 2. Step-by-Step Validation Checklist

### Step 1: Spatial Resampling
- **Method**: Resampled to exact **$1.0 \times 1.0 \times 1.0\,\text{mm}^3$ isotropic voxel grid**.
- **Image Interpolator**: `sitkBSpline` (3rd-order B-spline interpolation for T1, T1ce, T2, FLAIR).
- **Mask Interpolator**: `sitkNearestNeighbor` (preserves discrete integer label integrity).
- **Validation Result**: **PASSED**. Zero voxel grid distortion or aliasing artifacts observed.

### Step 2: N4 Bias Field Correction
- **Filter**: `SimpleITK.N4BiasFieldCorrectionImageFilter`.
- **Parameters**: `MaximumNumberOfIterations = [50, 50, 50, 50]`, `ConvergenceThreshold = 0.001`.
- **Validation Result**: **PASSED**. Corrected low-frequency intensity non-uniformity caused by 3T RF coils across brain parenchyma.

### Step 3: Intensity Normalization
- **Method**: Z-score normalization strictly within brain parenchyma mask ($\mu = 0.0, \sigma = 1.0$, scaled by factor $100.0$).
- **Validation Result**: **PASSED**. Standardized intra-tumoral contrast across multi-sequence acquisitions.

### Step 4: Multi-Label Tumor Segmentation Alignment
- **Labels Present**: $1 = \text{Necrotic/Non-Enhancing Core (NET)}$, $2 = \text{Peritumoral Edema (ED)}$, $4 = \text{Enhance Tumor (ET)}$.
- **Whole Tumor Mapping**: Merged into binary Whole Tumor mask ($\text{WT} = \{1, 2, 4\}$) for shape, geometry, and global texture feature extraction.
- **Validation Result**: **PASSED**. Segmentation boundary aligned perfectly with structural T1ce/FLAIR tumor hyperintensities.

### Step 5: PyRadiomics Extraction Test
- **Engine**: PyRadiomics v3.0.1 (IBSI compliant).
- **Parameters**: Fixed bin width $= 25$, Z-score normalization enabled, bin count bounded ($16 \le N_{\text{bins}} \le 128$).
- **Features Extracted**: 733 features across Shape3D, FirstOrder, GLCM, GLRLM, GLSZM, NGTDM, GLDM, and Wavelet sub-bands.
- **Validation Result**: **PASSED**. 100% finite float array output; zero NaN or Inf values generated.

---

## 3. Edge Cases & Failed Sample Analysis

1. **Non-Contiguous Segmentation Masks**: 5 candidates out of 630 exhibited isolated single-voxel segmentation islands detached from the primary tumor mass. Mesh surface reconstruction in PyRadiomics throws topological boundary warnings on such masks.
   - *Handling Rule*: Flagged and excluded in Stage 8 of the QC pipeline (`results/v3/upenn_qc.csv`).
2. **Dynamic Range Extremes**: High-intensity vascular structures on T1ce caused extreme tail values in GLCM calculations.
   - *Handling Rule*: Bounded by brain-mask Z-score clipping at $[-3.0, +3.0]$ standard deviations prior to binning.
