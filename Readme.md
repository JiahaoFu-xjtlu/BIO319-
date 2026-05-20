# Unsupervised Multi-scale Manifold Learning Reveals a GC-rich m6Am-like Subtype within Human m6A Candidate Sites

This repository contains the code and result figures for my final-year project on unsupervised subtype discovery from human m6A candidate sites.

The project starts from m6A site data downloaded from the m6AConquer database. Sequence windows of different lengths were extracted using the `BSgenome.Hsapiens.UCSC.hg38` package in R, converted into four-channel one-hot encoded matrices, and then used for deep learning-based representation learning.

The main analysis applies VAE-based latent representation learning, UMAP projection, and HDBSCAN clustering to identify hidden sequence-derived subtypes. An isolated subtype, Cluster 0, was discovered through unsupervised clustering and was later interpreted as an m6Am-like subtype through post-hoc biological annotation, including motif analysis, transcription start site proximity, H3K4me3 enrichment, SHAP attribution, and Core-versus-Dropout comparison.

Importantly, m6Am labels were not used during model training, UMAP projection, or HDBSCAN clustering.

---

## Project Overview

The overall workflow is:

```text
m6AConquer m6A site data
→ sequence extraction using BSgenome.hg38 in R
→ one-hot encoding in R
→ deep learning model training
→ VAE latent feature extraction
→ UMAP projection and HDBSCAN clustering
→ motif logo generation
→ m6AConquer omics feature download
→ cluster-feature merging in R
→ Random Forest and SHAP analysis
→ downstream biological and statistical analyses

The approximate reproduction order is:

train
→ umap
→ generate logo
→ shap
→ further analysis
Data Sources

The m6A candidate site data were downloaded from m6AConquer:

http://rnamd.org/m6aconquer

Main site file:

m6A_OrthogonallyValidatedSites_Combined_hg38.rds

The genomic and epigenetic feature file was also downloaded from m6AConquer:

m6Aconquer_omicsFeaturesV2_hg38.csv

The RDS site file was used to obtain candidate m6A positions. The omics feature file was later merged with clustering results for biological annotation and SHAP-based interpretation.

Depending on data redistribution restrictions, the original m6AConquer data files may need to be downloaded directly from the m6AConquer website.

Repository Structure

The repository is organized by analysis stage and sequence scale.

.
├── 21bp_raw_data/              # Raw 21 bp one-hot sequence analysis
├── 21_ae_new/                  # 21 bp deterministic autoencoder baseline
├── 21bp_vae_new/               # 21 bp VAE analysis and downstream results
├── 51bp_vae_new/               # 51 bp VAE analysis
├── 101bp_vae_new/              # 101 bp VAE analysis
├── all_length_analysis/        # Cross-scale comparison and tracking analysis
├── average_silhouette_score/   # Silhouette score comparison across models
├── Rstudio/                    # R scripts for sequence extraction, one-hot encoding, and feature merging
└── README.md

The main analysis folders contain the Python scripts and generated figures for each model or sequence scale.

Main Analysis Steps
1. Sequence Extraction and One-hot Encoding

The m6A candidate sites were first processed in R.

Sequence windows were extracted at three scales:

21 bp
51 bp
101 bp

The extraction was performed using the human hg38 reference genome through:

BSgenome.Hsapiens.UCSC.hg38

Each sequence was then converted into a four-channel one-hot encoded matrix:

A = [1, 0, 0, 0]
C = [0, 1, 0, 0]
G = [0, 0, 1, 0]
U = [0, 0, 0, 1]

These one-hot matrices were used as input for deep learning models.

2. Model Training

Different representation models were trained and compared, including:

raw one-hot representation,
deterministic autoencoder baseline,
variational autoencoder representation.

The VAE model was used as the main representation learning framework because it produced a continuous latent manifold suitable for downstream subtype discovery.

3. UMAP and HDBSCAN Clustering

After model training, VAE latent features were extracted and analysed using UMAP and HDBSCAN.

The same UMAP and HDBSCAN parameters were used across the 21 bp, 51 bp, and 101 bp VAE analyses.

UMAP parameters:

n_neighbors = 50
min_dist = 0.0
n_components = 2
metric = "euclidean"
random_state = 42

HDBSCAN parameters:

min_cluster_size = 1000
min_samples = 50
metric = "euclidean"
cluster_selection_method = "eom"

At the 21 bp scale, the VAE-HDBSCAN workflow identified 11 motif-associated subtypes. Cluster 0 formed a small isolated subtype.

4. Motif Logo Generation

Sequence logos were generated for HDBSCAN-derived clusters to examine their local motif patterns.

Cluster 0 showed a distinct CAG-centered motif, which differed from many broader DRACH-like motif patterns in the background clusters.

5. Feature Merging

The clustering results were merged in R with the m6AConquer omics feature file:

m6Aconquer_omicsFeaturesV2_hg38.csv

This merged feature table was used for downstream biological annotation and interpretation.

The main post-hoc annotation features included:

transcription start site proximity,
H3K4me3 status,
GC content,
other genomic and epigenetic features from m6AConquer.
6. SHAP Attribution and Downstream Analysis

A Random Forest surrogate model was trained after unsupervised clustering to approximate the cluster boundary between Cluster 0 and background sites.

SHAP attribution was then used to interpret which external biological features characterized the isolated subtype.

Additional downstream analyses included:

H3K4me3 enrichment analysis,
TSS proximity analysis,
cross-scale tracking,
Core-versus-Dropout comparison,
GC-content comparison between stable and dropout sequences.
Key Results

The main findings are:

The 21 bp VAE-HDBSCAN analysis identified 11 motif-associated subtypes.
Cluster 0 formed a small isolated subtype containing 2,739 sites.
Cluster 0 showed a distinct CAG-centered motif.
Cluster 0 was substantially closer to the 5′ transcript end than background sites.
Cluster 0 showed strong H3K4me3 enrichment.
Cross-scale tracking identified 1,806 conserved Core sites and 933 Dropout sites.
SHAP attribution suggested that promoter-associated features and GC-related features characterize the isolated subtype.
Core sequences showed higher flanking GC content than Dropout sequences.

Together, these results support the interpretation of Cluster 0 as an m6Am-like subtype within the original human m6A candidate pool.

How to Reproduce

A simplified reproduction workflow is:

Step 1: Prepare data in R

Download the m6A site data from m6AConquer.

Use R scripts to:

extract 21 bp, 51 bp, and 101 bp sequences using BSgenome.Hsapiens.UCSC.hg38;
convert sequences into one-hot encoded matrices;
prepare input files for Python model training.
Step 2: Train models

Run the model training scripts in the corresponding folders.

Example:

python train_vae.py
Step 3: Run UMAP and HDBSCAN

Example:

python umap_hdbscan_vae.py
Step 4: Generate motif logos

Example:

python generate_logos.py
Step 5: Merge clustering results with omics features

Use R to merge cluster labels with:

m6Aconquer_omicsFeaturesV2_hg38.csv
Step 6: Run SHAP analysis

Example:

python shap_feature_analysis.py
Step 7: Run further feature analysis

Example:

python futher_feature_analysis.py

Note: Some script names may differ slightly between folders depending on the analysis stage. Please follow the folder-specific scripts and comments.

Requirements

Main Python packages include:

numpy
pandas
matplotlib
scikit-learn
umap-learn
hdbscan
torch
shap
scipy
logomaker

R packages used in preprocessing include:

BSgenome.Hsapiens.UCSC.hg38
Biostrings
GenomicRanges
dplyr
data.table

Install Python dependencies with:

pip install -r requirements.txt

If requirements.txt is not provided, install the packages above manually.

Notes on Interpretation

This project follows a discovery-first design.

The isolated subtype was first identified by unsupervised sequence-derived topology. Biological features such as motif structure, TSS proximity, H3K4me3 enrichment, and GC content were only used after clustering for post-hoc interpretation.

Therefore, Cluster 0 should be described as an m6Am-like subtype rather than a pre-labelled m6Am class.

The computational results support the presence of a GC-rich, promoter-associated m6Am-like subtype within the m6A candidate pool. Further experimental validation would be required to confirm modification identity and biochemical mechanism at single-site resolution.

Author

Jiahao Fu
Final Year Project
Supervisor: Dr. Zhen Wei
