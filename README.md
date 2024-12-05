**NLP-Disaster-Tweets-Classification**

### README Description:

# NLP Disaster Tweets Classification

This repository contains the second assignment for **ORIE 5750: Applied Machine Learning**, focusing on text classification using the NLP Disaster Tweets dataset. The goal is to predict whether a given tweet represents a real disaster or not. The project incorporates preprocessing, feature extraction, and multiple classification models.

---

## Project Overview

The assignment involves building a machine learning pipeline for natural language processing (NLP). Using the Disaster Tweets dataset, the workflow includes:
1. **Data Preprocessing**:
   - Lowercasing, lemmatization, and stopword removal.
   - Cleaning text by removing URLs, punctuation, and non-ASCII characters.
   - Implementing Bag of Words (BoW) and N-gram feature extraction.

2. **Modeling**:
   - Logistic Regression with and without L1/L2 regularization.
   - Bernoulli Naive Bayes for binary classification.

3. **Evaluation**:
   - Comparing models using F1-scores on training, validation, and test datasets.
   - Inspecting logistic regression weight vectors to identify impactful features.

4. **Submission**:
   - Predictions for test data formatted for Kaggle submission.

---

## Results

- Logistic Regression with L2 regularization achieved the best performance on test data:
  - **Public Kaggle Score**: **0.78761**
- Bernoulli Naive Bayes performed comparably, demonstrating the simplicity and effectiveness of generative models for text classification.

---

## Technologies Used
- **Python**:
  - Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`
- **Machine Learning**:
  - Logistic Regression (L1 and L2 regularization)
  - Bernoulli Naive Bayes
- **Natural Language Processing**:
  - Feature extraction using CountVectorizer (BoW, N-grams).

---

## Repository Contents
- **`Assignment_2.ipynb`**: Main notebook with preprocessing, model training, and evaluation.
- **`train.csv`**: Training dataset with labeled tweets.
- **`test.csv`**: Test dataset for Kaggle submission.
- **`sample_submission.csv`**: Template for submission to Kaggle.
- **`submission.csv`**: Final predictions for test data.
- **`Homework_2_ORIE_5750__AML_Merged.pdf`**: Detailed write-up of the assignment.

---

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/NLP-Disaster-Tweets-Classification.git
   cd NLP-Disaster-Tweets-Classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
   ```

3. **Run the Notebook**:
   Open `Assignment_2.ipynb` in Jupyter Notebook and execute cells sequentially to reproduce the results.

---

## Authors
- **Rishab Jain** (`rj424`)
- **Shalom Otieno** (`soo26`)
