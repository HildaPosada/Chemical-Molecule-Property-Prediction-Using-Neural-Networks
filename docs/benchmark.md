# Reproducible checkpoint evaluation

Run `python scripts/benchmark.py` from the repository root after installing requirements. This evaluates the committed checkpoint on the saved SQLite test split and trains a logistic regression and class-prior baseline using only saved training rows. Invalid structures are excluded and counted.

On 308 valid test molecules, the committed checkpoint achieved 85.06% accuracy and 0.8975 ROC-AUC. Logistic regression achieved 86.69% accuracy and 0.8880 ROC-AUC. The class-prior baseline achieved 76.30% accuracy. The neural model is not uniformly better than the simpler baseline.

Twelve canonical molecular structures overlap between training and test. These are retrospective random-split results. They do not establish performance on scaffold-disjoint molecules or a prospective external dataset. The saved scaler was built with scikit-learn 1.6.1; reproduce with the pinned environment when comparing exact numbers.

Detailed metrics, checkpoint checksum, predictions and incorrect predictions are saved under `experiments/benchmark/`. Scores are uncalibrated. No clinical validity is claimed.
