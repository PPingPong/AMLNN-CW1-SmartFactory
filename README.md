# 6FTC2175 AMLNN — Coursework 1: SmartFactory Predictive Quality Control

Classical machine learning vs a feed-forward neural network vs a convolutional neural network, for detecting faulty production at a fictional manufacturer, SmartFactory.

| Part | Data | Model |
|---|---|---|
| A | AI4I 2020 Predictive Maintenance (UCI id 601), 10,000 sensor records | Random Forest and feed-forward NN (MLP) |
| B | Real-life industrial dataset of casting product (Kaggle): 7,348 images, plus a second export of 1,300 images used only as an external test | CNN trained from scratch (3 versions) |
| C | — | Comparison, cost-based thresholds and deployment recommendation |

## Repository contents
| File | Purpose |
|---|---|
| `CW1_SmartFactory_PartsABC.ipynb` | The full notebook (Parts A, B and C) with all outputs from the reported run |
| `requirements.txt` | Library versions the notebook was executed with |
| `results/` | CSV tables written by the notebook (test metrics, CV, split, robustness, second export, cost analysis) and `environment.txt` |

## Data (downloaded automatically, not stored in this repository)
- **AI4I 2020:** fetched with `ucimlrepo` (UCI id 601). If the download fails, the notebook asks you to upload `ai4i2020.csv` from the UCI/Kaggle page.
- **Casting images:** fetched with `kagglehub` from `ravirajsinh45/real-life-industrial-dataset-of-casting-product` (public; no API key needed).

## How to reproduce
1. Open the notebook in Google Colab (*File → Upload notebook*).
2. Select *Runtime → Change runtime type → T4 GPU*.
3. Select *Runtime → Run all*. No manual steps are needed.
4. Run time: the reference run took 46 minutes on a 2-core CPU without a GPU (Part A ≈ 12 min, Part B ≈ 34 min). With a GPU, Part B is much faster.

## Evaluation design
- **Part A:** a stratified 70/15/15 split. Model family and hyper-parameters are chosen by cross-validation on the training set; early stopping, model choice and decision thresholds use the validation set. Both models are scored on the same held-out test rows, once.
- **Part B:** the author's train/test folders are **not** used, because 47% of the author's test images have a copy of the same casting in the author's training folder (64 are byte-identical).
  - All 7,348 images are pooled and copies are grouped with a rotation/flip-aware perceptual hash. The data contains no casting IDs.
  - The groups are then split 70/15/15 *before any training*, and no copy crosses the splits.
  - The architecture and threshold are chosen on validation; the grouped test set is used once.
  - The second export is never used for tuning.
- **Part C:** cost ratios (20:1 for sensors, 50:1 for images) are **illustrative assumptions**. Thresholds are always chosen on validation data, and a sensitivity analysis covers ratios from 1 to 100.

## Reproducibility notes
- One random seed (`RANDOM_STATE = 42`) is used everywhere. The data splits, the grouping, the Random Forest and all scikit-learn results are bit-for-bit repeatable.
- TensorFlow training is not bit-for-bit deterministic by default, especially on a GPU, so MLP and CNN numbers can differ slightly from the reported run. Set `DETERMINISTIC = True` in the setup cell for repeatable runs, at some cost in speed. The seed-variance table (Step A7) shows how much the MLP varies.
- `results/environment.txt` records the library versions and whether a GPU was used.

## Main findings (reported run)
- **Part A:** the Random Forest beat the MLP on the same 1,500 test rows: F1 0.929 vs 0.843 and PR-AUC 0.937 vs 0.890 (McNemar exact p = 0.012). It was also better in all 5 paired CV folds (PR-AUC 0.917 ± 0.040 vs 0.734 ± 0.114) and identical across seeds. With 51 test failures, the recall intervals overlap (Wilson 95%: 0.79–0.96 vs 0.72–0.92).
- **Part B:** on the leak-free grouped test set (1,020 images), the selected CNN (V2, augmentation + dropout) reached F1 0.968 and MCC 0.922, missing 35 of 630 defects. The baseline V1 overfits, and V3 (global average pooling) underfits within 20 epochs. On the independent second export, F1 fell to 0.847 (recall 0.77), and brighter lighting halves recall. Generalisation to new imaging set-ups is the main open risk.
- **Recommendation:** deploy the Random Forest on the sensors with a human-review band. Pilot the CNN in shadow mode with a recall-oriented threshold until it is validated on SmartFactory's own images with part IDs. Do not deploy the MLP.

## Data sources
- Matzka, S. (2020) AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository, id 601.
- Dabhi, R. (2020) Real-life industrial dataset of casting product. Kaggle.
