# Sentiment Analysis on Book Reviews

This project implements a complete machine learning pipeline to classify book reviews as positive or negative. Using a labeled dataset of customer reviews, I followed the full ML life cycle from exploratory data analysis to model training, evaluation, and optimization.

## 📌 Technical Highlights

- **Data Source:** Book reviews dataset containing review text and binary sentiment labels.
- **Data Preprocessing:**  
  - Checked for missing values and verified class balance between positive and negative reviews.  
  - Split dataset into **training (75%)** and **testing (25%)** sets using `train_test_split` with a fixed random seed for reproducibility.
- **Feature Engineering:**  
  - Applied **TF-IDF vectorization** via `TfidfVectorizer` to transform raw text into numerical feature vectors.  
  - Captured term importance while reducing the influence of overly frequent words.
- **Modeling:**  
  - Implemented **Logistic Regression** from `scikit-learn` for binary sentiment classification.  
  - Evaluated using **ROC AUC** to measure discriminative ability beyond accuracy.
- **Results & Insights:**  
  - Achieved high ROC AUC score, showing strong capability to distinguish between positive and negative reviews.  
- **Tools & Libraries:** Python, Pandas, NumPy, Scikit-learn
