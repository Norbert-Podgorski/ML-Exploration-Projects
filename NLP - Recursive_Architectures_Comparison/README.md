# Recursive Architectures Comparison

## Description
This project focuses on comparing various recursive neural network architectures, using the AG News Subset dataset.

### Dataset:
The AG News Subset is a collection of news articles, categorized into four classes:
- World
- Sports
- Business
- Sci/Tech
  
Each article in the dataset consists of a title and a description, which can be used for tasks such as text classification. The goal is to predict the correct category (label) of a given news article based on its content.

### Project Goal:
The primary objective of this project is to compare the performance of different recurrent neural network architectures on the AG News Subset dataset.
The focus lies in analyzing how well these architectures perform in text classification, particularly in terms of their ability to generalize and avoid overfitting.

The architectures being compared are:
- <b>Simple RNN</b>
- <b>GRU (Gated Recurrent Unit)</b>
- <b>LSTM (Long Short-Term Memory)</b>
- <b>Bidirectional LSTM</b>

Each of these models will be trained and evaluated with the same dataset to determine their strengths and weaknesses. In particular, the comparison will focus on:
- Overfitting: Identifying which architectures are more prone to overfitting and how they can be mitigated.
- Generalization: Analyzing how well the models generalize to unseen data.
- Accuracy and Loss: Comparing the training and validation performance metrics of each architecture.

