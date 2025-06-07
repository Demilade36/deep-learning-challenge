# Alphabet Soup Deep Learning Challenge

## Overview

This project involves building a deep learning model for the nonprofit organization **Alphabet Soup**. The goal is to create a binary classification model that predicts whether an applicant will be successful if funded by Alphabet Soup, using data from more than 34,000 historical funding records.

We used **TensorFlow** and **Keras** to design and train a neural network and improved its performance through multiple optimization strategies.

---

## Project Structure
deep-learning-challenge/
│
├── AlphabetSoupCharity.ipynb                # Initial model development
├── AlphabetSoupCharity_Optimization.ipynb   # Optimized deep learning model
├── AlphabetSoupCharity.h5                   # Saved initial model weights
├── AlphabetSoupCharity_Optimization.h5      # Saved optimized model weights
├── charity_data.csv                         # Original dataset
├── README.md                                # Project documentation
└── AlphabetSoupCharity_report.md            # Final written report

---

## Key Objectives

- Preprocess the dataset using `pandas` and `StandardScaler`.
- Design and train a neural network for binary classification.
- Evaluate model performance on test data.
- Optimize the model to exceed 75% prediction accuracy.
- Summarize findings in a report.

---

## Technologies Used

- Python
- Pandas
- TensorFlow / Keras
- Scikit-learn
- Google Colab

---

## Model Summary

| Model Version     | Accuracy | Loss   | Hidden Layers | Notes                              |
|-------------------|----------|--------|----------------|------------------------------------|
| Initial Model     | 73.08%   | 0.5558 | 2              | Base model, under target accuracy  |
| Optimized Model   | 78.55%   | 0.4999 | 3              | Added layers, tuned neurons, removed noise |

---

## Optimization Techniques Used

- Added a third hidden layer for deeper learning
- Tuned number of neurons in each layer
- Used `ReLU` activations in hidden layers, `Sigmoid` in output
- Reduced training epochs to prevent overfitting
- Dropped `STATUS` column to remove noise
- Included `NAME` column to test for useful patterns

---

## Future Improvements

- Try Random Forest or XGBoost for comparison
- Analyze feature importance to improve interpretability
- Test additional categorical binning strategies
- Use AutoML for architecture tuning

---

## How to Run

1. Clone this repository:
2. Open the notebooks in Google Colab or Jupyter Notebook:
- `AlphabetSoupCharity.ipynb` for the base model
- `AlphabetSoupCharity_Optimization.ipynb` for the optimized model

3. Make sure `charity_data.csv` is in your workspace.

4. Run all cells to reproduce the results.

---
