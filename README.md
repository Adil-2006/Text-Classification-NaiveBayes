# Text Classification Using Naive Bayes

## Overview

This project implements a Text Classification system using the Multinomial Naive Bayes algorithm to classify SMS messages as Spam or Ham (Not Spam).

## Dataset

SMS Spam Collection Dataset

* Total Messages: 5572
* Ham Messages: 4825
* Spam Messages: 747

## Technologies Used

* Python
* Pandas
* NLTK
* Scikit-Learn
* Matplotlib
* Seaborn

## Workflow

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Feature Extraction
5. Train-Test Split
6. Naive Bayes Training
7. Model Evaluation
8. Spam Prediction

## Model Performance

* Accuracy: 97.49%
* Precision: 100.00%
* Recall: 81.33%
* F1 Score: 89.71%

## Output Files

* confusion_matrix.png
* model.pkl
* vectorizer.pkl

## Run Project

```bash
pip install -r requirements.txt
python main.py
```

## Sample Prediction

Input:

Congratulations! You have won a free iPhone.

Output:

Prediction: SPAM
