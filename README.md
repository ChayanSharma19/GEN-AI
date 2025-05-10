Fake News Detection

Author

Chayan Sharma


Description

This repository features a streamlined fake news classification project, developed as part of the **GEN AI – NLP coursework** at **VIT Bhopal University**. The system leverages **Logistic Regression** to distinguish between fake and real news articles, achieving an accuracy of approximately **92%**. While the foundation of the project is inspired by more complex BERT and LSTM-based models, this version is crafted for clarity and ease of use.

What's Inside?

`my_fake_news_detection.ipynb` – The main Jupyter Notebook with code for preprocessing, model training, and evaluation.

`confusion_matrix.png` – Graphical representation of the model’s confusion matrix.

`explanation.md` – A concise technical overview of the implementation (for Phase 2 submission).

`report.tex` – LaTeX-formatted final report (Phase 3 submission).


⚙️ Getting Started

#### 1. Clone the Repository

git clone https://github.com/ChayanSharma19/GEN-AI.git

2. Set Up Your Environment

Make sure you have **Python 3.8 or above**. Then install the required dependencies:

pip install pandas nltk scikit-learn numpy seaborn matplotlib

3. Add the Dataset

Due to file size limitations, the dataset isn't included here.

Download the "Fake and Real News Dataset" from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).
Place the files as shown:

```
GENai_Project_22BCY10231/
└── data/
    ├── Fake.csv
    └── True.csv
```

4. Run the Notebook

Start Jupyter Notebook and open the file:

jupyter notebook my_fake_news_detection.ipynb

Run all cells (from `Cell > Run All`) to preprocess the data, train the model, and review the output.


Key Results
Accuracy: ~92%

Classification Metrics:

  -Fake News: Precision: 0.91 | Recall: 0.93 | F1-score: 0.92
  -Real News*: Precision: 0.93 | Recall: 0.91 | F1-score: 0.92
  -Output Visualization**: Confusion matrix saved in `confusion_matrix.png`
  -Example Prediction**:

  
 Notes

  -This implementation simplifies advanced deep learning workflows into a more accessible pipeline using traditional ML.
  -All experiments were conducted on a personal laptop for feasibility and portability in academic settings.


Acknowledgments
•	Dataset sourced from Kaggle: Fake and Real News Dataset.
•	Inspired by a downloaded fake news detection project that used BERT and LSTM models, whose workflows (Figures 1 and 2) informed this simplified implementation.

