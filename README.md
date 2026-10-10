# 📱 PCA on Mobile Price Data

**Research question:** *Can PCA reduce the dimensionality of mobile feature data while preserving important information?*

A machine-learning mini-project using the Kaggle **Mobile Price Classification** dataset. It walks through the full workflow: data loading, preprocessing, EDA, statistical analysis, PCA, model evaluation, performance curves and interpretation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/PCA_Mobile_Project_Colab.ipynb)

---

## 📂 Repository contents

| File | Description |
|---|---|
| `PCA_Mobile_Project_Colab.ipynb` | The complete project notebook (run this in Google Colab) |
| `train.csv` | 2000 labelled phones (20 features + `price_range`) |
| `test.csv` | 1000 unlabelled phones (used only to produce predictions) |
| `README.md` | This file |

## 📊 Dataset

- **Source:** [Mobile Price Classification on Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- **Features (20):** `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`
- **Target:** `price_range` (0 = low, 1 = medium, 2 = high, 3 = very high); 500 phones per class.

---

## ▶️ How to run in Google Colab (step by step)

1. **Open the notebook**
   - Click the **Open in Colab** badge above, **or**
   - go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub tab** → paste this repo's URL → choose `PCA_Mobile_Project_Colab.ipynb`.
2. **Get the data files** on your computer: download `train.csv` and `test.csv` from this repo (open each file → *Download raw file*) or from Kaggle.
3. **Run the cells in order** with **Shift + Enter** (or **Runtime → Run all**).
4. At cell **A1 – Upload your files**, click **Choose files** and select **both** `train.csv` and `test.csv`. Wait until the upload shows 100%.
5. Let the remaining cells finish. Read the **Observation** text under each output.
6. The last code cell creates `test_predictions.csv` and downloads it automatically.

> ⚠️ If you restart the runtime, upload the CSV files again (Colab clears uploaded files).

No installation is needed: Colab already has pandas, numpy, scipy, matplotlib, seaborn and scikit-learn.

### Run locally instead (optional)
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter
jupyter notebook PCA_Mobile_Project_Colab.ipynb
```
Change the paths `/content/train.csv` and `/content/test.csv` to `train.csv` and `test.csv`, and skip the `files.upload()` and `files.download()` lines.

---

## 🧭 What the notebook does

| Section | Content |
|---|---|
| **A. Data loading** | Shape, columns, data types, head/tail, `info()` |
| **B. Preprocessing** | Missing values, duplicates, binary vs numeric columns, fixing impossible zeros (`px_height`, `sc_w`), train/validation split, standard scaling |
| **C. EDA** | Bar and pie chart, histograms, boxplots, correlation heatmap, scatter and pair plot, feature dendrogram |
| **D. Statistics** | Range, IQR, MAD, variance, standard deviation, Pearson correlation, outliers (IQR and Z-score) |
| **E. PCA** | Explained variance, scree plot, loadings, 2-D projection, where `ram` lands |
| **F. Evaluation** | Logistic regression on all features vs PCA features: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, GridSearchCV tuning |
| **G. Curves** | Reconstruction error, accuracy vs components, learning curve, K-Means (elbow and silhouette), predictions for `test.csv` |
| **H. Interpretation** | Answers to the assignment questions |

## 🔑 Key results

| Setting | Variance kept | Validation accuracy |
|---|---|---|
| All 20 features | 100% | 96.3% |
| PCA, 17 components | about 94% | 96.0% |
| PCA, 10 components | about 63% | 56.0% |
| PCA, 2 components | about 16% | 25.5% (≈ random guess) |

- 14 / 16 / 18 components are needed to keep 80% / 90% / 95% of the variance.
- Tuned model (PCA with 17 components + Logistic Regression, C = 10): **97.25%** validation accuracy.
- K-Means on the PCA space gives a silhouette score of only about 0.09, so the price classes do not form natural clusters.

## ✅ Conclusion

PCA reduces this dataset only slightly. Most mobile features are nearly uncorrelated, so there is little redundancy to remove, and about 17 of 20 components are needed to keep the predictive information. Aggressive reduction loses it, because `ram`, the strongest price predictor, is independent of the other features and appears only in a later component (PC6). PCA ranks components by variance, not by usefulness for the target.

**Limitations:** PCA is unsupervised and linear, and its components are harder to interpret. `test.csv` has no labels, so evaluation uses an 80/20 split of `train.csv`.
**Improvements:** supervised reduction (LDA or feature selection), non-linear models, or t-SNE/UMAP for visualization.

## 🛠️ Tools

Python · pandas · NumPy · SciPy · Matplotlib · Seaborn · scikit-learn

## 📄 Credit

Dataset by Abhishek Sharma on Kaggle. Please check the dataset page for its license terms.
