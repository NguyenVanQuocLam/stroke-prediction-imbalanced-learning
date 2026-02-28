\# Stroke Prediction with Imbalanced Learning \& Fairness Analysis



\## Overview



This project investigates stroke prediction using Logistic Regression

under severe class imbalance conditions.



We compare:



\- Baseline (no balancing)

\- SMOTE-NC

\- CTGAN oversampling



Additionally, fairness analysis is performed across:



\- Gender

\- Residence Type



---



\## Methods



\- Stratified Train-Test Split

\- Logistic Regression (class\_weight='balanced')

\- SMOTE-NC

\- CTGAN

\- Fairness Metrics:

&nbsp; - TPR

&nbsp; - FPR

&nbsp; - ROC-AUC by group



---



\## How to Run



```bash

pip install -r requirements.txt

python src/train.py

