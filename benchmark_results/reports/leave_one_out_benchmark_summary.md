# Leave-One-Out Benchmark Summary

## Datasets and Scope
- **Datasets Analyzed**: globins, thioredoxins, kinases
- **Total Rows**: 309
- **Methods Evaluated**: bary_init, infer_rand, infer_bary

## Per-Method Performance Trends
Summary statistics for `delta_qlocal`, `delta_qglobal`, and `insertion_time`.

| method     |   delta_qlocal_mean |   delta_qlocal_median |   delta_qlocal_std |   delta_qlocal_min |   delta_qlocal_max |   delta_qglobal_mean |   delta_qglobal_median |   delta_qglobal_std |   delta_qglobal_min |   delta_qglobal_max |   insertion_time_mean |   insertion_time_median |   insertion_time_std |   insertion_time_min |   insertion_time_max |
|:-----------|--------------------:|----------------------:|-------------------:|-------------------:|-------------------:|---------------------:|-----------------------:|--------------------:|--------------------:|--------------------:|----------------------:|------------------------:|---------------------:|---------------------:|---------------------:|
| bary_init  |         -0.00147432 |           -0.00103273 |          0.0125508 |         -0.0630345 |          0.0462468 |          5.99328e-05 |            1.55571e-05 |          0.00210775 |         -0.00582866 |           0.0142141 |            0.00444283 |               0.0017216 |             0.01185  |            0.0004797 |            0.0854914 |
| infer_bary |         -0.00445482 |           -0.00295427 |          0.0150782 |         -0.103053  |          0.0477279 |         -0.00142323  |           -0.0013218   |          0.00216197 |         -0.00651742 |           0.0101777 |            0.543417   |               0.381211  |             0.29686  |            0.283831  |            1.20835   |
| infer_rand |         -0.00431654 |           -0.00295427 |          0.0150353 |         -0.103053  |          0.0477279 |         -0.00141855  |           -0.00130798  |          0.00216268 |         -0.00651926 |           0.0101768 |            0.55591    |               0.383442  |             0.295711 |            0.269738  |            1.32271   |

## Radial-Bin Trends
Performance variations depending on the protein's radial position in the full map.

| method     | radius_bin   |   delta_qlocal_mean |   delta_qlocal_median |   delta_qglobal_mean |   delta_qglobal_median |   insertion_time_mean |   insertion_time_median |
|:-----------|:-------------|--------------------:|----------------------:|---------------------:|-----------------------:|----------------------:|------------------------:|
| bary_init  | Center       |          0.00196336 |          -0.00103273  |          6.15132e-06 |           -0.000124515 |            0.00256917 |              0.0018249  |
| bary_init  | Mid          |         -0.00467185 |          -0.00155224  |         -0.000170589 |            0.000103582 |            0.00548567 |              0.00178005 |
| bary_init  | Periphery    |         -0.00181557 |          -0.000755688 |          0.000345818 |            3.04602e-05 |            0.00532875 |              0.0015707  |
| infer_bary | Center       |         -0.00107493 |          -0.002128    |         -0.000661159 |           -0.000775956 |            0.543735   |              0.375608   |
| infer_bary | Mid          |         -0.00991063 |          -0.00407299  |         -0.00210325  |           -0.00214924  |            0.573202   |              0.389377   |
| infer_bary | Periphery    |         -0.00247831 |          -0.00284216  |         -0.00152768  |           -0.00153837  |            0.513305   |              0.382183   |
| infer_rand | Center       |         -0.00107493 |          -0.002128    |         -0.00066119  |           -0.000775992 |            0.561243   |              0.382488   |
| infer_rand | Mid          |         -0.00949064 |          -0.00395214  |         -0.00208662  |           -0.00214833  |            0.571646   |              0.391013   |
| infer_rand | Periphery    |         -0.00247938 |          -0.00284216  |         -0.00153013  |           -0.0015391   |            0.534684   |              0.376094   |

## Correlation Results
Spearman rank correlations between `full_map_radius` and delta metrics.

| method     |   corr_radius_vs_delta_qlocal |   corr_radius_vs_delta_qglobal |
|:-----------|------------------------------:|-------------------------------:|
| bary_init  |                    -0.0144203 |                      0.0521021 |
| infer_rand |                    -0.166696  |                     -0.433148  |
| infer_bary |                    -0.159272  |                     -0.43026   |

## Outlier Summary
A total of **46** outliers were identified using the Tukey 1.5 IQR rule for `delta_qlocal`.
Detailed overlapping matrices and raw outlier rows are available in the generated CSV files.