# Demand Forecasting with Machine Learning  
**Author:** Atharva Shailesh Anchalwar  

This repository contains a comprehensive forecasting framework developed for **energy demand**, **Walmart retail sales**, and **Rossmann store sales** datasets. The goal is to build a **robust, scalable, and domain-adaptable forecasting system** using classical statistical models, tree-based machine learning models, and deep learning architectures—combined through **dynamic weighting and stacked ensembles**.

The full report is included in the repository.  

## 📌 Project Overview

Demand forecasting is essential for industries like **energy management** and **retail**, where accurate future predictions directly influence:

- Resource allocation  
- Inventory and production planning  
- Cost minimization & operational efficiency  
- Customer satisfaction  

This project evaluates three diverse datasets (Energy, Walmart, Rossmann) to demonstrate how **machine learning + feature engineering + ensembles** can systematically improve forecast accuracy across different domains.

---

## 📂 Datasets

### **Energy Demand**
- Hourly data with strong **seasonality** and **temporal dependencies**
- Suitable for sequence models (LSTM/GRU)
- Weather features (temperature, humidity) were added  

### **Walmart Sales**
- Driven by **promotions**, **holidays**, and store-specific effects  
- Rich categorical data → required extensive encoding  

### **Rossmann Sales**
- Contains **competition distance**, store attributes, and promotions  
- Required **per-store modeling** due to high variability  

## 🧠 Models Used

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **SARIMAX** | Captures trend + seasonality | Poor performance on complex retail data |
| **XGBoost** | Strong on nonlinear relationships | Requires engineered features |
| **Random Forest** | Stable, robust baseline | Can underfit long-term trends |
| **LSTM** | Captures sequential dependencies | Needs heavy preprocessing |
| **GRU** | Faster version of LSTM | Slightly lower accuracy |

## 🛠️ Feature Engineering

A major performance driver across all datasets:

- Lag features (daily/weekly/monthly)  
- Rolling means  
- Holiday/seasonality flags  
- Store-level and weather features  
- Handling high- vs low-cardinality categorical data  

---

## 🔧 Implementation Summary

### **1. Individual Models**
- **SARIMAX**: Seasonal order (0,1,0,12) for yearly seasonality  
- **XGBoost**: Tuned (max_depth=3, η=0.1, n_estimators=100)  
- **Random Forest**: 100 trees, depth=5  
- **LSTM**: 50 hidden units, dropout=0.2  
- **GRU**: Tested for energy demand datasets  

### **2. Ensemble Framework**
#### **Dynamic Weighting**
Weights are assigned as:  
\[
w_i = \frac{1}{MSE_i}
\]
Higher-performing models influence predictions more—illustrated on **page 12, Fig. 6**.  

#### **Stacked Generalization**
- Linear Regression meta-model  
- Learns complementary strengths of SARIMAX, XGBoost, RF, and LSTM  
- Removes over-reliance on any single model  

---

## Results

### **Overall Performance by Dataset**
(As reported in Table 2, page 8)  

#### **Energy Demand**
- **XGBoost:** 2.65% MSE  
- **Ensemble:** 3.01%  
- LSTM performance dropped after GRU introduction

#### **Walmart Sales**
- **XGBoost:** 6.12%  
- **Ensemble:** 4.33% (best)

#### **Rossmann Sales**
- **Ensemble:** 0.57% (exceptionally strong)  
- SARIMAX completely failed (Fig. 5, p. 11)  

### Visual Comparisons
Figures from the report show:
- Improved Walmart predictions after tuning (Fig. 1, p. 8)  
- Smooth Rossmann ensemble forecasts (Fig. 2, p. 9)  
- GRU vs ensemble energy plots (Fig. 3 & Fig. 4, p. 10)  
- SARIMAX failures (Fig. 5, p. 11)

---

### Dataset Insights
- **Energy**: Seasonality-heavy → LSTM most effective  
- **Walmart**: Promotions dominate → XGBoost/RF excel  
- **Rossmann**: Competition + store attributes → ensemble best  

### General Insights
1. **Feature engineering** drives accuracy more than model choice  
2. **Dynamic weighting** improves robustness  
3. **No one-size-fits-all model** for forecasting  
4. **Clustering stores** can massively reduce training cost (future extension)  
5. **SARIMAX unreliable** for complex or sparse retail data  

---
