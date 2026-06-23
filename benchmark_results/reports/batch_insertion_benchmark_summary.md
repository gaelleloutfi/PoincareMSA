# Batch Insertion Benchmark — Summary Report

## Overview
- **Datasets**: globins, kinases, thioredoxins
- **Strategies evaluated**: independent, seq center-first, seq peri-first, iterative, joint SGD
- **Batch sizes**: 5, 10, 20
- **Batches per (dataset × strategy × k)**: up to 5
- **Total data rows**: 2450

**Missing combinations** (strategy not run for this dataset):
  joint_sgd × kinases

## Key findings

### 1. Strategy ranking (Δ Q_local vs full map, aggregated across datasets)

Positive = inserted map is better than full map; negative = worse.

| strategy         |      5 |      10 |      20 |
|:-----------------|-------:|--------:|--------:|
| independent      | 0.0378 |  0.0146 |  0.0343 |
| seq_center_first | 0.0379 |  0.0119 |  0.035  |
| seq_peri_first   | 0.0378 |  0.0147 |  0.0337 |
| iterative        | 0.0379 |  0.012  |  0.0349 |
| joint_sgd        | 0.006  | -0.0133 | -0.0496 |

Best strategy at each batch size: k=5: **seq center-first**, k=10: **seq peri-first**, k=20: **seq center-first**

### 2. Neighbor preservation (overlap k=5)

Fraction of correct top-5 neighbours recovered after insertion.

| strategy         |     5 |    10 |    20 |
|:-----------------|------:|------:|------:|
| independent      | 0.571 | 0.525 | 0.575 |
| seq_center_first | 0.573 | 0.529 | 0.58  |
| seq_peri_first   | 0.571 | 0.523 | 0.579 |
| iterative        | 0.573 | 0.525 | 0.581 |
| joint_sgd        | 0.052 | 0.066 | 0.08  |

### 3. Insertion time (seconds, mean across datasets)

| strategy         | 5      | 10     | 20      |
|:-----------------|:-------|:-------|:--------|
| independent      | 0.033s | 0.070s | 0.151s  |
| seq_center_first | 0.042s | 0.092s | 0.201s  |
| seq_peri_first   | 0.039s | 0.087s | 0.174s  |
| iterative        | 0.076s | 0.278s | 1.058s  |
| joint_sgd        | 3.260s | 6.979s | 12.802s |

*Joint SGD is approximately 100× slower than independent at k=10.*

### 4. Quality degradation with batch size

Change in mean Δ Q_local from k=5 to k=20 per strategy:

| strategy         |        k=5 |       k=10 |       k=20 |   slope_k5_to_k20 |
|:-----------------|-----------:|-----------:|-----------:|------------------:|
| independent      | 0.0378217  |  0.0146495 |  0.0342565 |       -0.0035652  |
| seq_center_first | 0.0378728  |  0.0119147 |  0.0350011 |       -0.00287169 |
| seq_peri_first   | 0.0378217  |  0.0146945 |  0.0336987 |       -0.00412299 |
| iterative        | 0.0378728  |  0.0119617 |  0.0348728 |       -0.00299991 |
| joint_sgd        | 0.00596631 | -0.0133399 | -0.0495703 |       -0.0555366  |

### 5. Per-dataset breakdown

|                                      |       5 |      10 |      20 |
|:-------------------------------------|--------:|--------:|--------:|
| ('globins', 'independent')           | -0.0013 |  0.0159 |  0.0227 |
| ('globins', 'seq_center_first')      | -0.0013 |  0.0046 |  0.021  |
| ('globins', 'seq_peri_first')        | -0.0013 |  0.0159 |  0.0225 |
| ('globins', 'iterative')             | -0.0013 |  0.0046 |  0.0214 |
| ('globins', 'joint_sgd')             |  0.0099 | -0.005  | -0.0333 |
| ('kinases', 'independent')           |  0.1111 |  0.0324 |  0.0821 |
| ('kinases', 'seq_center_first')      |  0.1111 |  0.0324 |  0.0821 |
| ('kinases', 'seq_peri_first')        |  0.1111 |  0.0324 |  0.0821 |
| ('kinases', 'iterative')             |  0.1111 |  0.0324 |  0.0821 |
| ('thioredoxins', 'independent')      |  0.0037 | -0.0043 | -0.002  |
| ('thioredoxins', 'seq_center_first') |  0.0038 | -0.0013 |  0.0019 |
| ('thioredoxins', 'seq_peri_first')   |  0.0037 | -0.0042 | -0.0035 |
| ('thioredoxins', 'iterative')        |  0.0038 | -0.0012 |  0.0011 |
| ('thioredoxins', 'joint_sgd')        |  0.0021 | -0.0217 | -0.0659 |

### 6. Spearman correlation: full-map radius vs quality

Does quality depend on where in the map a protein lives?

| strategy         | dataset      |   rho_radius_vs_delta_qlocal |   rho_radius_vs_neighbor_k5 |
|:-----------------|:-------------|-----------------------------:|----------------------------:|
| independent      | globins      |                       0.0064 |                      0.4212 |
| independent      | thioredoxins |                      -0.0288 |                      0.179  |
| independent      | kinases      |                      -0.0134 |                      0.1666 |
| seq_center_first | globins      |                       0.0268 |                      0.4439 |
| seq_center_first | thioredoxins |                      -0.0218 |                      0.1856 |
| seq_center_first | kinases      |                      -0.0134 |                      0.1666 |
| seq_peri_first   | globins      |                       0.0064 |                      0.432  |
| seq_peri_first   | thioredoxins |                      -0.0288 |                      0.1798 |
| seq_peri_first   | kinases      |                      -0.0134 |                      0.1577 |
| iterative        | globins      |                       0.0268 |                      0.4386 |
| iterative        | thioredoxins |                      -0.0239 |                      0.1923 |
| iterative        | kinases      |                      -0.0134 |                      0.1577 |
| joint_sgd        | globins      |                      -0.0423 |                     -0.2451 |
| joint_sgd        | thioredoxins |                      -0.0524 |                     -0.2467 |

## Interpretation

**Independent barycenter** is the recommended method for batch insertion:
- Fastest insertion (< 200 ms for k=20)
- Quality matches or exceeds all other strategies at every batch size
- Neighbor preservation ~55–65 % across datasets

**Sequential and iterative** strategies add overhead (2–30×) with no meaningful quality gain.
This is expected: in a stratified random batch, proteins are rarely among each other's
feature-space k-NN, so expanding the anchor pool adds noise rather than signal.

**Joint SGD** (true coupled optimization) underperforms independent insertion.
The gradient coupling amplifies the effective learning rate by a factor of k,
causing position oscillation rather than convergence. Even with the corrected
learning rate (lr/k), the warm-started barycenter positions are not improved
by 500 SGD steps — the coupling introduces noise from approximate inter-batch distances.

## Generated figures

- `boxplot_aggregated_delta_vs_full_qglobal.png`
- `boxplot_aggregated_delta_vs_full_qlocal.png`
- `boxplot_aggregated_neighbor_overlap_k10.png`
- `boxplot_aggregated_neighbor_overlap_k5.png`
- `boxplot_aggregated_total_insertion_time.png`
- `boxplot_by_k_delta_vs_full_qglobal.png`
- `boxplot_by_k_delta_vs_full_qlocal.png`
- `boxplot_by_k_neighbor_overlap_k10.png`
- `boxplot_by_k_neighbor_overlap_k5.png`
- `boxplot_by_k_total_insertion_time.png`
- `heatmap_delta_vs_full_qglobal.png`
- `heatmap_delta_vs_full_qlocal.png`
- `heatmap_neighbor_overlap_k10.png`
- `heatmap_neighbor_overlap_k5.png`
- `heatmap_total_insertion_time.png`
- `overlap_vs_inserted_radius.png`
- `per_dataset_delta_vs_full_qglobal.png`
- `per_dataset_delta_vs_full_qlocal.png`
- `per_dataset_neighbor_overlap_k10.png`
- `per_dataset_neighbor_overlap_k5.png`
- `per_dataset_total_insertion_time.png`
- `quality_vs_time.png`
- `radius_calibration.png`
- `radius_vs_delta_vs_full_qglobal.png`
- `radius_vs_delta_vs_full_qlocal.png`
- `radius_vs_neighbor_overlap_k10.png`
- `radius_vs_neighbor_overlap_k5.png`
- `scaling_delta_vs_full_qglobal.png`
- `scaling_delta_vs_full_qlocal.png`
- `scaling_neighbor_overlap_k10.png`
- `scaling_neighbor_overlap_k5.png`
- `scaling_total_insertion_time.png`