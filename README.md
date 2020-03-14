Yelp Reviews Sentiment Analysis
==============================

Empirical analysis of various ML/AL algorithms and text pre-processing techniques on text classification task on the Yelp review dataset

## Data

- Restaurants
- Toronto
- +10 reviews

## Metrics

- **sklearn.metrics.classification_report**
  - Precision
  - Recall
  - F-1
  - Both Macro and Micro averages

## Baseline for Dataset and Pre-Processing Tests

- Model: Multinomial Naive Bayes (**sklearn.naive_bayes.MultinomialNB**)
  - Default parameters (alpha = 1.0, fit_prior = True, class_prior = None)

- Data
  - 270k training data, 90k testing data
  - Class ratios: 9%, 10%, 18%, 33%, 30%
- Pre-processing (both training and testing)
  - Unigram
  - Word tokenized, bag-of-words (**CountVectorizer**)
  - Lower-cased

## Dataset Tests

- Sample size
  - Downsample 270k trainig data to 1k, 60k, 130k, 200k
- Class balance
  - Downsample all classes to 25k each, total of 125k

## Pre-Processing Tests

- Number of ngrams
  - Unigrams + Birams, Unigrams + Bigrams + Trigrams
- Removes non-words/numbers (**nltk.tokenize.RegexpTokenizer**)
- Removes stopwords (**nltk.corpus.stopwords**)
- Stemming (**nltk.stem.porter.PorterStemmer**)
- Lemmatization (**nltk.stem.wordnet.WordNetLemmatizer**)
- POS Tagging (**nltk.post_tag**)
- Feature generated using TF-IDF (Term Frequency-Inverse Document Frequency) (**TFidfVectorizer**)

## Baseline for Algorithmic Tests

- Model: Multinomial Naive Bayes
  - Best set of pre-processing tests
- Ten-fold cross-validated

## Algorithms

- Maximum Entropy
- Ordinal Regression
- Logistic Regression
- SVM
- XGBoost
- LSTM
- BERT

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
