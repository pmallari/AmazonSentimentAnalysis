# Sentiment Analysis on Amazon Reviews

This repository contains a PyTorch implementation of a Deep Bag of Words (DBOW) method and a PyTorch reimplementation of the UCLMR (https://arxiv.org/pdf/1707.03264.pdf).

The goal is accurately predict the sentiment of each amazon review. The amazon review is scraped by Brian McMahan and Delip Rao in their [website](dl4nlp.info). The reviews are positive if they are rated with more than 3 stars and negative otherwise. Each review is separated based on their title and their content.

For benchmarking, we used the Text Blob library to predict whether a review is positive or negative. Text Blob is a popular Python library for sentiment analysis. We took the sentiment polarity of the title and content separately and took the mean to get the score. If a score is greater than or equal to 0, it is positive. If it is less than 0, it is negative. The Text Blob library had an accuracy of **71.67%**.

The DBOW method uses the Bag of Words representation of the entire title and content based on GloVe embeddings (https://nlp.stanford.edu/projects/glove/). The embeddings are concatenated along with the cosine similarity of the TFIDF scores of the title and the content. The embeddings and cosine similarity are unrolled and fed into a neural network. The network's accuracy on the test set is **85.15%**.

The UCLMR method uses the term frequencies of the top 5,000 words instead of the word embeddings. This implementation uses only 3,000 words because of memory limitations. The feature's are unrolled and fed into a neural network. The network's accuracy on the test set is **86.58%**.
