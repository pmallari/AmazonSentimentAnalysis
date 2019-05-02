# Sentiment Analysis on Amazon Reviews

This repository contains a PyTorch implementation of a Deep Bag of Words (DBOW) method and a PyTorch reimplementation of the UCLMR (https://arxiv.org/pdf/1707.03264.pdf).

For benchmarking, we will use the Text Blob library to predict whether a review is positive or negative. Text Blob is a popular Python library for sentiment analysis. The Text Blob library scored a **71.67%**.

The DBOW method uses the Bag of Words representation of the entire title and content based on GloVe embeddings (https://nlp.stanford.edu/projects/glove/). The embeddings are concatenated along with the cosine similarity of the TFIDF scores of the title and the content. The feature is fed into a neural network. The network's accuracy on the test set is **85.15%**.

The UCLMR method uses the term frequencies of the top 5,000 words instead of the word embeddings. This implementation uses only 3,000 words because of memory limitations. The feature is fed into a neural network. The network's accuracy on the test set is **86.58%**.
