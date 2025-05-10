Fake News Detection

Author
Chayan Sharma

Description

This repository contains a fake news detection project developed for the GEN AI course NLP assignment at VIT BHOPAL University. The project classifies news articles as fake or real using natural language processing (NLP) and machine learning. It uses Logistic Regression with the Kaggle Fake and Real News Dataset, achieving approximately 92% accuracy. The implementation is inspired by advanced workflows for clickbait and fake news article detection from a downloaded project, adapted for simplicity and accessibility.
The repository includes:
•	my_fake_news_detection.ipynb: Jupyter Notebook with the implementation code.
•	confusion_matrix.png: Visualization of the model's performance.
•	explanation.md: Brief summary of the implementation (Phase 2 deliverable).
•	report.tex: Final report in LaTeX (Phase 3 deliverable).

Instructions

1.	Clone the Repository:
2.	Set Up Environment:
o	Ensure Python 3.8+ is installed.
o	Install required libraries:
pip install pandas nltk scikit-learn numpy seaborn matplotlib
3.	Download the Dataset:
o	The Kaggle Fake and Real News Dataset (Fake.csv, True.csv) is not included due to file size limits. Download it from Kaggle.
o	Place the files in a data/ subfolder:
o	GENai_Project_22BCY10231/data/Fake.csv
GENai_Project_22BCY10231/data/True.csv
4.	Run the Notebook:
o	Open my_fake_news_detection.ipynb in Jupyter Notebook:
jupyter notebook my_fake_news_detection.ipynb
o	Execute all cells (Cell > Run All) to preprocess the data, train the model, and generate results.

Output
•	Accuracy: 92%
•	Classification Report:
o	Fake: Precision: 0.91, Recall: 0.93, F1-score: 0.92
o	True: Precision: 0.93, Recall: 0.91, F1-score: 0.92
•	Visualization: Confusion matrix saved as confusion_matrix.png.
•	Sample Prediction: "Government claims new policy boosts economy, lacks evidence." → Fake

Notes
•	This implementation is a simplified version tailored for accessibility, using Logistic Regression and the Kaggle dataset, distinct from the original project’s BERT and LSTM approaches.
•	Results were obtained by running the notebook on a standard laptop, ensuring practicality for the GEN AI course submission.
Acknowledgments
•	Dataset sourced from Kaggle: Fake and Real News Dataset.
•	Inspired by a downloaded fake news detection project that used BERT and LSTM models, whose workflows (Figures 1 and 2) informed this simplified implementation.

