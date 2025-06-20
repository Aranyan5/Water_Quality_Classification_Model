# Water Quality Prediction using Machine Learning

-By Aranyan T
-SRMIST Trichy

A machine learning pipeline that predicts **Water Quality Index (WQI)** and classifies water samples into quality categories such as *Poor*, *Marginal*, *Fair*, and *Good* using physicochemical parameters. Built for real-world environmental monitoring and sustainability analysis.

---

## 🔍 Project Overview

This project processes a raw water quality dataset, computes WQI, and uses **Random Forest Classifier** to categorize water samples based on pollutant concentrations like O₂, NH₄, NO₃, NO₂, PO₄, SO₄, Cl⁻, BSK5, and Suspended Solids.

**Key features:**
- Cleans and imputes missing or invalid data  
- Calculates WQI using weighted indices  
- Classifies water into quality classes using supervised ML  
- SHAP explainability for feature impact  
- Saves trained model for reuse

---

## 🧪 Technologies Used

- **Python 3.11**
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `shap`, `joblib`, `imbalanced-learn`
- `RandomForestClassifier` (sklearn)

---

## 📁 Dataset

- File: `water Quality prediction data set.csv`
- Fields include:  
  `O2`, `NH4`, `NO3`, `NO2`, `PO4`, `SO4`, `CL`, `BSK5`, `Suspended`, `date`, `id`

---

## 🧠 ML Pipeline Overview

1. **Load + Clean Data**
   - Drop duplicates and handle negative values
   - Forward fill + mean imputation

2. **Feature Engineering**
   - Compute Water Quality Index (WQI)
   - Classify quality using WQI bins

3. **Train/Test Split**
   - Stratified split with rare class filtering

4. **Model Training**
   - Standard scaling + Random Forest classification

5. **Model Evaluation**
   - Accuracy, classification report, confusion matrix

6. **Model Explainability**
   - SHAP summary plots per class

7. **Model Persistence**
   - Save scaler and classifier using `joblib`

---

## 📊 Results

- **Test Accuracy:** >85% (varies slightly by data)
- **Top Influential Features (SHAP):**  
  O₂, NH₄, BSK5, NO₂, Suspended Solids

---

## 📂 Model Output

Saved as:  
```bash
rf_wqi_quality_classifier_optimized.joblib
