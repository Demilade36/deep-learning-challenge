# Alphabet Soup Deep Learning Challenge Report

## Overview of the Analysis

Alphabet Soup is a nonprofit organization that funds various ventures. This analysis aims to build a binary classification model that predicts whether an applicant will be successful if funded. We used deep learning techniques to train and optimize a neural network model using historical funding data.

---

## Results

### Data Preprocessing

- **Target Variable**:  
  `IS_SUCCESSFUL` – Indicates whether the funded organization was successful.

- **Feature Variables**:  
  All other columns excluding identification and status-related fields, such as:  
  `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`, etc.

- **Removed Variables**:  
  - `EIN` and `NAME` – Identification columns, not predictive.  
  - `STATUS` – Removed during optimization to reduce noise.

---

### Compiling, Training, and Evaluating the Model

#### Initial Model (AlphabetSoupCharity.ipynb)
- **Layers**:  
  2 hidden layers  
- **Activation Functions**:  
  ReLU (hidden layers), Sigmoid (output layer)  
- **Performance**:  
  - Accuracy: **73.08%**  
  - Loss: **0.5558**

#### Optimized Model (AlphabetSoupCharity_Optimization.ipynb)
- **Model Architecture**:
  - Hidden Layer 1: 80 neurons (ReLU)
  - Hidden Layer 2: 40 neurons (ReLU)
  - Hidden Layer 3: 10 neurons (ReLU)
  - Output Layer: 1 neuron (Sigmoid)

- **Performance**:
  - Accuracy: ✅ **78.55%** (Target: 75%)
  - Loss: **0.4999**

#### Steps Taken to Improve Performance:
- ✔ Added a third hidden layer to enhance pattern recognition
- ✔ Tuned neuron counts for feature abstraction
- ✔ Used ReLU activation to support faster and more stable learning
- ✔ Reduced the number of epochs (to 80) to reduce overfitting
- ✔ Dropped `STATUS` column to minimize data noise
- ✔ Included `NAME` column to test for potential hidden signals

---

## Summary and Recommendations

The deep learning analysis conducted for Alphabet Soup successfully produced a binary classification model capable of predicting the likelihood of applicant success with funding. The initial model achieved 73.08% accuracy, slightly below the target of 75%. However, through thoughtful optimization strategies—such as adding a third hidden layer, tuning neuron counts, dropping noisy features, and managing training epochs—the optimized model improved significantly and achieved an accuracy of **78.55%**, with a reduced loss of **0.4999**.

This outcome demonstrates that deep learning, particularly neural networks, can effectively handle complex, non-linear relationships in structured datasets with categorical and numerical features. The ability to extract patterns from a variety of organizational and financial attributes is key to supporting Alphabet Soup in making informed funding decisions.

### Recommendation:
While the current deep learning model performs well, a tree-based model such as **Random Forest** or **XGBoost** could be tested. These models often handle categorical data and class imbalances effectively and may provide more interpretability with feature importance scores.

---

