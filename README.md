# Kaggle Mushroom Classification using Random Forest
This repository holds an attempt to to classify mushrooms as edible or poisonous using Random Forests, based on categorical features from the UCI Mushroom Dataset found on Kaggle (https://www.kaggle.com/datasets/uciml/mushroom-classification/data).
# Overview
This project classifies mushrooms as edible or poisonous based on its physical attributes. Each observation describes a mushroom sample across 22 categorical features (odor, cap color, bruises, etc), and the target variable is its edibility. The approach in this repository formulates the problem as a classification task. We preprocess the data by dropping a constant feature (veil-type), then one-hot encoding the features (X) and label encoding the target (y). We used 5-fold cross-validation, and further split the training set (80%) into training and validation sets (87.5%/12.5%) for hyperparameter tuning. The test fold (20%) was used for final evaluation. A Random Forest Classifier was trained and evaluated using accuracy and weighted F1-score, achieving perfect performance (100%). Key predictive features include odor, gill-size, and gill-color.
## Summary of Work Done

### Data

- **Type**: Categorical CSV data (tabular)
- **Input**: 21 categorical features per mushroom sample 
- **Output**: Binary label:  edible (0) and poisonous (1)
- **Size**: 8124 instances 

### Preprocessing / Clean-up

- Dropped ‘veil-type', and ‘stalk-root' (missing values)
- One-hot encoded all categorical features using `ColumnTransformer` and `OneHotEncoder`
- Label-encoded target feature


### Data Visualization

- Bar plots and frequency tables showing feature distributions (e.g., odor, cap color) across classes
- Conditional probability tables:
  - **P(Class | Feature Category)** : proportion of edible vs. poisonous mushrooms within each category of a feature.
  - **P(Feature Category | Class)** : distribution of feature categories within class.
<img width="275" alt="Percentage of Edible and Poisonous Mushrooms" src="https://github.com/user-attachments/assets/09732a52-7a29-44b0-9804-58ca55010e62" />
<img width="275" alt="odor vs class (percent)" src="https://github.com/user-attachments/assets/7d2ce71a-f5dd-42ff-b5fa-3c356c596e68" />
<img width="275" alt="AD_4nXfdfI9d91CQWmgNCwxLqgqwIcHnbXUpDFOk72FC3OeQ0AJ22LMB2hDdm1uXGYIZEq7ngMfKX2ZYAg7rjaIF9TKgfMkydtaZhET-w4Sx4lTIoKD3z2sCPtjI" src="https://github.com/user-attachments/assets/392d801b-4cc5-4df7-989a-845142221d63" />
<img width="275" alt="gill-color vs class (percent)" src="https://github.com/user-attachments/assets/785d4f14-3e80-4894-8200-770759b7ac70" />


From these graphs, we can see that odor, gill-color and gill-size are good  features to explore since they show a good distinction of class within each category. 
**Odor** is especially one of the most discriminative features for predicting mushroom class.
- Mushrooms with odor = almond (a), anise (l), none (n) are nearly 100% edible in this dataset.
- Mushrooms with odor = creosote (c), fishy (y), foul (f), musty (m), pungent (p), spicy (s), are associated with 100% poisonous mushrooms.

This means that the odor feature can serve as a reliable predictor for classifying mushrooms, especially coupled with the other features.

### Problem Formulation

- Input: One-hot encoded categorical features (20)
- Output: Binary edible/poisonous class
  - **Split**: 5-fold cross-validation: each fold uses 80% of data for training (split into 87.5% train, 12.5% validation) and 20% as the test set
- Model: Random Forest Classifier (sklearn)
- Hyperparameters: max_depth, n_estimators tuned via validation set
- Metrics: accuracy and weighted F1 score

## Training

- Environment: Jupyter Notebook on local machine
- sklearn for model training and metrics
- Training duration: Fast (seconds per fold)
- K-fold CV used to maximize generalizability
- Best hyperparameters selected based on validation accuracy
  - max_depth = 10
  - n_estimators = 10
- Feature importances averaged across folds

## Random Forest Performance

**Classification Report**
| Class         | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (edible)   | 1.00      | 1.00   | 1.00     | 846     |
| 1 (poisonous)| 1.00      | 1.00   | 1.00     | 778     |
| **Accuracy** |           |        | **1.00** | 1624    |
| **Macro avg**| 1.00      | 1.00   | 1.00     | 1624    |
| **Weighted avg** | 1.00  | 1.00   | 1.00     | 1624    |


**Confusion Matrix**

|               | Predicted: Edible (0) | Predicted: Poisonous (1) |
|---------------|-----------------------|---------------------------|
| Actual: Edible (0)   | 846                   | 0                         |
| Actual: Poisonous (1)| 0                     | 778                       |

## Conclusions

- **Odor** is the most important predictor of mushroom class (especially foul odor for poisonous)
- **Gill-size** and **gill-color** also contribute significantly
- Random Forests perform very well on this structured categorical data


 
