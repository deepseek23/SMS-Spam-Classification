# SMS Spam Classification

A machine learning project that classifies SMS messages as **Spam** or **Ham** using a trained text classification model and a simple **Streamlit** web app.

The notebook covers the full pipeline from data loading and text preprocessing to model training and evaluation. The final trained model and vectorizer are saved in the `model/` directory and used by the Streamlit application in `app.py`.

## Project Highlights

- Cleaned and prepared SMS text data for machine learning.
- Applied text preprocessing such as lowercasing, tokenization, stopword removal, and stemming.
- Built and evaluated multiple classification models.
- Saved the best trained model and TF-IDF vectorizer for reuse.
- Built an interactive Streamlit app for live spam prediction.

## What I Used

- **Python**
- **Jupyter Notebook** for exploration, preprocessing, training, and evaluation
- **Pandas** for data handling
- **NumPy** for numerical operations
- **Matplotlib** and **Seaborn** for visualization
- **NLTK** for text preprocessing
- **Scikit-learn** for feature extraction and classification
- **Streamlit** for the web interface
- **Pickle** for saving the trained model and vectorizer

## Dataset

This project uses an SMS spam dataset containing labeled messages:

- `ham` for normal messages
- `spam` for unwanted or promotional messages

The dataset is stored in the `data/` folder.

## Machine Learning Workflow

1. Load the dataset
2. Clean the text data
3. Transform text using tokenization and stemming
4. Convert text to numerical features with TF-IDF
5. Train and evaluate classification models
6. Save the final model and vectorizer
7. Use Streamlit to make predictions from user input

## Model Files

The trained artifacts are stored here:

- `model/model.pkl`
- `model/vectorizer.pkl`

These files are loaded by the Streamlit app when making predictions.

## Model Selection

Several classifiers were evaluated on the same TF-IDF features. The strongest results from the notebook were:

| Model | Accuracy | Precision |
| --- | ---: | ---: |
| Extra Trees Classifier | 0.9758 | 0.9748 |
| SVC | 0.9729 | 0.9741 |
| Random Forest | 0.9720 | 1.0000 |
| Multinomial Naive Bayes | 0.9516 | 1.0000 |
| XGBoost | 0.9720 | 0.9658 |

Although tree-based models and SVC achieved slightly higher accuracy, the final deployed model was **Multinomial Naive Bayes** because it is a better fit for this problem:

- SMS spam detection uses sparse TF-IDF text features, which Naive Bayes handles very well.
- Precision matters more than raw accuracy here, because a false spam prediction can incorrectly flag a normal message.
- Multinomial Naive Bayes achieved **1.0000 precision** in the notebook, meaning it avoided false positives in the tested split.
- It is lightweight, fast to load, and ideal for a simple Streamlit app.

In short, the chosen model balances strong performance with practical deployment advantages.

## Project Structure

```text
SMS-Spam-Classification/
├── app.py
├── data/
│   └── spam.csv
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── notebook/
│   └── notebook.ipynb
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

Start the app from the project root:

```bash
streamlit run app.py
```

Then paste an SMS message into the text box and click **Predict** to see whether it is classified as **Spam** or **Ham**.

## How It Works

The app uses the same preprocessing pipeline as the notebook:

- convert text to lowercase
- remove non-alphanumeric characters
- tokenize the message
- remove English stopwords
- apply stemming
- vectorize the cleaned text with TF-IDF
- predict the class using the saved model

## Why This Model Works Well

SMS spam classification is a classic text classification problem with high-dimensional, sparse features. The final model performs well because it learns directly from word-frequency patterns rather than needing complex feature engineering.

The selected Multinomial Naive Bayes model is especially effective because:

- it is designed for count-based and TF-IDF-like features,
- it is less computationally expensive than many ensemble models,
- it generalizes well on small-to-medium text datasets,
- and it produces a simple, reliable spam vs ham decision that is easy to deploy.

This makes it a strong choice for a lightweight production-style demo where reliability and speed matter.

## Notes

- Make sure the `model/` directory contains both `model.pkl` and `vectorizer.pkl`.
- The notebook is included for transparency and reproducibility of the training workflow.

## Author

Created as part of a machine learning project focused on SMS spam detection.