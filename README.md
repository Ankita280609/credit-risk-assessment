# Project 10: Credit Risk Prediction & Agentic Lending Advisor
## From Financial Risk Modeling to Automated Lending Advice

### Project Overview
This project involves the design and implementation of an **AI-driven credit analytics system** that evaluates borrower credit risk and evolves into an agentic AI lending decision support assistant.

- **Milestone 1:** Classical machine learning techniques applied to historical borrower data to predict default probability, assess loan risk, and identify key financial risk drivers.
- **Milestone 2:** In progress...

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Random Forest, Scikit-Learn |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit |
| **LLMs (M2)** | Open-source models or Free-tier APIs |
| **Language** | Python 3.13 |
| **Hosting** | Streamlit Community Cloud |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Credit Risk Prediction (Mid-Sem)
**Objective:** Identify high-risk loan applicants using historical borrower data, focusing on classical ML pipelines *without LLMs*.

### Key Deliverables

- **Problem Understanding & Business Context**  
  Clear definition of the financial risk problem, business impact of loan defaults, and the objective of building a data-driven credit risk prediction system.

- **System Architecture & Preprocessing Pipeline Design**  
  Modular ML architecture using Scikit-Learn `Pipeline` and `ColumnTransformer`, including median imputation, standardization, one-hot encoding (`handle_unknown='ignore'`), class balancing, and leakage-free training–inference consistency.

- **Working Deployed Application with UI**  
  Live Streamlit web application for real-time borrower risk scoring:  
  https://ankita-credit-risk.streamlit.app/

- **Model Performance Evaluation Report**
Comprehensive evaluation using Accuracy, ROC-AUC, F1-score (Macro & Weighted), and Confusion Matrix with class-wise precision and recall analysis.
https://drive.google.com/file/d/17iUsRswDZUvzuUFtVy4husvHpzSgS6-P/view?usp=drive_link
  
- **Demo Video**

  
**Results Summary:**
| Model | Accuracy | ROC-AUC Score | Primary Strength |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8151 | 0.8696 | Interpretability & Regulatory Compliance |
| Random Forest | 0.9331 | 0.9288 | Superior Accuracy & Non-linear Capture |

---

### Team
| Member | Role | Responsibilities |
| :--- | :--- | :--- |
| Himani Pinjani | Data Engineer | Dataset sourcing, cleaning, and exploratory analysis |
| Ankita Thakur | ML Engineer | Model training, evaluation, and pipeline design |
| Anshu Yadav | UI Developer & Deployment Lead | Streamlit interface and cloud deployment |
| Farhana Pervin | Documentation Lead | Performance report and project documentation |

---

### Repository Structure
```
CREDIT_RISK_APP/
├── app.py                        # Streamlit UI and real-time inference logic
├── train.py                      # Model training, evaluation, and serialization
├── logistic_model.pkl            # Serialized Logistic Regression pipeline
├── random_forest_model.pkl       # Serialized Random Forest pipeline
├── credit_risk_dataset (4).csv   # Raw labelled borrower dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```
