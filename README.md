Yelp Reviews Sentiment Analysis
==============================

https://arxiv.org/abs/2004.13851

- 350,000 Yelp reviews on 5,000 restaurants
- Ablation study on text preprocessing techniques
  - For machine learning models, we find that using binary bag-of-word representation, adding bi-grams, imposing minimum frequency constraints and normalizing texts have positive effects on model performance
  - For deep learning models, we find that using pre-trained word embeddings and capping maximum length often boost model performance
- Using macro F1 score as our comparison metric, we find simpler models such as Logistic Regression and Support Vector Machine to be more effective at predicting sentiments than more complex models such as Gradient Boosting, LSTM and BERT
