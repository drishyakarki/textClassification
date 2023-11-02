# Text Classification Using Pytorch

## Description

In this repo, I have used **AG_NEWS dataset** which has the following four labels:

```
1 : World
2 : Sports
3 : Business
4 : Sci/Tec
```

The provided code defines a Python class named TextClassificationModel for a text classification task. It uses word embeddings to represent text data and a neural network for classification.

## Key concepts

**Text Classification Model**: The class is designed for text classification tasks, such as sentiment analysis or topic classification.

**Word Embeddings**: The model uses word embeddings, which are dense vector representations of words, to capture semantic and contextual information about words. These embeddings are used to convert text data into a format suitable for deep learning.

**EmbeddingBag Layer**: The nn.EmbeddingBag layer is used to handle the word embeddings. It converts a sequence of word indices (text) into a dense vector representation. The vocab_size is the size of the vocabulary, embed_dim is the dimensionality of the word embeddings, and sparse indicates whether the input is sparse (typically, it's set to False).

**Linear Layer**: The model contains a linear layer (nn.Linear) that performs the classification. It takes the embedded text data and maps it to the number of output classes (num_class). This layer will output the classification scores for each class.

**Initialization**: The model's weights for the embedding and linear layers are initialized with specific values using the init_weights method.

**Forward Method**: The forward method defines the forward pass of the model. It takes text data and their offsets as input. The text is first embedded using the EmbeddingBag layer, and then the embedded representation is passed through the linear layer to obtain classification scores.

*This model is designed to be used in text classification tasks, where the goal is to map input text data to specific categories or labels. It leverages word embeddings to capture the meaning of words in the input text, which is crucial for understanding and classifying text data.*
