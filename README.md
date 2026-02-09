# Fake News Detection using NLP and Transformer Models

Binary text classification system comparing traditional machine learning (TF-IDF + Logistic Regression) against transformer-based DistilBERT for fake news detection.

## 🎯 Project Overview

- **Dataset**: 45,000+ news articles
- **Models**: Logistic Regression vs. DistilBERT
- **Best Performance**: 97.66% accuracy (Logistic Regression), DistilBERT with 0.0202 evaluation loss
- **Technologies**: Python, scikit-learn, Transformers (Hugging Face), Pandas, NumPy

## 📊 Results

### Logistic Regression Performance
- **Accuracy**: 97.66%
- **Precision**: 96.63%
- **Recall**: 98.29%
- **F1 Score**: 97.45%

### DistilBERT Performance
- **Evaluation Loss**: 0.0202
- High accuracy confirmed by confusion matrix and low loss
- Superior contextual understanding compared to TF-IDF baseline

## 🛠️ Technologies Used

- Python 3.x
- scikit-learn
- Transformers (Hugging Face)
- Pandas & NumPy
- Matplotlib/Seaborn (for visualization)

## 🚀 Key Features

- Complete NLP preprocessing pipeline (tokenization, lemmatization, stopword removal)
- Comparative analysis of traditional ML vs. transformer models
- Evaluation of computational efficiency vs. performance trade-offs
- Applicable to spam detection, sentiment analysis, bias detection

## 📁 Project Structure
├── nlp-midterm.ipynb    # Main Jupyter notebook
├── requirements.txt     # Python dependencies
└── README.md           # This file
## 💡 Use Cases

This pipeline can be adapted for:
- Political bias detection
- Sentiment analysis
- Email phishing detection
- AI-generated content identification

## 📝 Course Context

Developed as midterm project for Natural Language Processing course.

## 👤 Author

Meetali Mandhare - CS Student specializing in ML/AI
