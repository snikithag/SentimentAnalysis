# 📊 Sentiment Analysis using Machine Learning

This project performs sentiment analysis on textual data to classify sentiments as **positive**, **negative**, or **neutral** using various machine learning models.

---

## 🧠 Project Overview

The goal of this project is to analyze textual data (e.g., reviews, tweets, comments) and determine the sentiment expressed. The pipeline includes:
- Text preprocessing
- Feature extraction (TF-IDF / Count Vectorizer)
- Model training using classification algorithms
- Performance evaluation and visualization

---

## 🛠️ Tech Stack

- **Language:** Python
- **IDE:** Jupyter Notebook
- **Libraries:**
  - `pandas`, `numpy` for data handling
  - `sklearn` for ML models and evaluation
  - `nltk` / `re` for text preprocessing
  - `matplotlib`, `seaborn` for plotting

---

## 🚀 How to Run

1. Clone the repository or download the `.ipynb` file:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk
   ```

3. Open the notebook:
   ```bash
   jupyter notebook SentimentAnalysis.ipynb
   ```

4. Run each cell in order and view the results.

---

## 📈 Models Used

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

Each model's performance is evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 📌 Results

- Efficient classification of text into sentiment categories.
- Visualization of model performance and word frequency.
- Identification of the best-performing classifier based on metrics.

