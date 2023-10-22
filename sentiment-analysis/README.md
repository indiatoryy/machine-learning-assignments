# Sentiment Analysis Using Gated-Recurrent Neural Networks

## Background

Sentiment Analysis, also known as opinion mining, is a natural language processing task that involves determining the sentiment expressed in a piece of text, such as positive, negative, or neutral. In this assignment, the task is to classify movie reviews as positive or negative using Gated-Recurrent Neural Networks (GRUs). The IMDB dataset, consisting of 50,000 movie reviews labeled as positive or negative, is utilized for this purpose.

## Data Representation

To apply machine learning algorithms, text data needs to be converted into a numerical format. One-Hot Encoding and Word Embeddings are common techniques used for this purpose.

- **One-Hot Encoding:** Words are represented as binary vectors where each word is encoded as a vector of zeros, except for the index representing the word, which is marked with a 1. For example, "apple" might be represented as [1, 0, 0, 0] in a dictionary containing words [apple, orange, Milan, Rome].

- **Word Embeddings:** A dense vector representation of words where words with similar meanings are mapped to nearby points in space. This mapping is often performed using an Embedding Layer, which transforms words into dense vectors of fixed size.

## Architecture

- **Embedding Layer:** The input words are converted into dense vectors using an embedding layer. Each word is represented as a sequence of integers and mapped to dense vectors based on pre-trained word embeddings or learned during training.

- **Gated Recurrent Unit (GRU):** GRUs are a type of Recurrent Neural Network (RNN) designed to capture dependencies in sequential data. The GRU cell uses update and reset gates to control the flow of information, allowing it to capture long-range dependencies in the data.

- **Fully-Connected Layers:** The output from the GRU cell is fed into fully-connected layers with ReLU activation functions. These layers transform the learned features into predictions about the sentiment of the input text.

## Optimization Techniques:

- **SGD vs. Adam:** The model is trained using both Stochastic Gradient Descent (SGD) and Adam optimizer. Comparing these optimizers helps assess their impact on the model's performance.

- **Early Stopping:** Early stopping is implemented with a patience window of 5 epochs. Training is halted if the validation loss does not improve within this window, preventing overfitting.

## Assignment Tasks:

1. **Training with Adam:** Train the model using the Adam optimizer and compare its performance with SGD.

2. **Early Stopping Implementation:** Implement early stopping with a patience window of 5 epochs to prevent overfitting.

Understanding the implementation of these techniques and their impact on the model's performance is essential to building effective sentiment analysis systems.
