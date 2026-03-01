# 🏦 Intelligent Credit Risk Scoring System

## Overview
This project is an intelligent machine learning-based credit risk scoring framework integrated with a lending decision support system. It evaluates borrower profiles to predict the probability of loan default using supervised learning algorithms.

**Live Application:** https://ankita-credit-risk.streamlit.app/

## System Architecture & Models
Two models were trained and evaluated on an imbalanced dataset of 32,581 records:
* **Logistic Regression (Baseline):** Achieved an Accuracy of 81.51% and ROC-AUC of 0.8696.
* **Random Forest (Primary Engine):** Achieved an Accuracy of 93.31% and ROC-AUC of 0.9288.

A comprehensive preprocessing pipeline was implemented, including median imputation, StandardScaler, and OneHotEncoder.

## Setup Instructions
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ankita280609/credit-risk-assessment.git](https://github.com/Ankita280609/credit-risk-assessment.git)
   cd credit-risk-assessment
