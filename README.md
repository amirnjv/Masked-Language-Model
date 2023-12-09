# Masked Language Model (MLM) with Transformers

![Alt text](https://devopedia.org/images/article/254/4557.1579959350.gif)

## Introduction
This project showcase how we can build a Masked Language Model (MLM) using the Hugging Face Transformers. Masked Language Modeling (MLM) is a pre-training technique for deep learning models in NLP. It works by randomly masking a portion of the input tokens in a sentence and asking the model to **predict the masked tokens**. The focus is on utilizing the Bert Encoder Network, particularly its ability to predict masked words in a sentence based on contextual and semantic understanding.

## Bert Encoder Network
Before diving into the practical aspects, it's essential to understand the core component of our project: the Bert Encoder Network. Bert, short for Bidirectional Encoder Representations from Transformers, is renowned for its effectiveness in understanding the context and semantic meaning of words in sentences. The model's ability to predict masked words - words that are intentionally hidden during training - is a cornerstone of its design. This guide will explore how Bert processes input data and the theory behind its training methodology.

## Dataset and Preprocessing
The IMDB dataset, known for its extensive collection of movie reviews, serves as the primary dataset for this project. Preprocessing steps involve tokenizing the text using the `AutoTokenizer` from the Transformers library and segmenting it into manageable portions suitable for training with the DistilBERT model. DistilBERT, a streamlined version of Bert, is chosen for its balance between performance and efficiency.

## Training
The training phase involves masking parts of the dataset, an essential process for MLM. The project details include setting up training and evaluation dataloaders, initializing the DistilBERT model, configuring the optimizer, and utilizing an accelerator to enhance training efficiency. The methodology adheres to PyTorch standards, focusing on DataLoader utilization, model optimization, learning rate adjustments, and evaluation metrics.

## Testing
Post-training, the project demonstrates the model's capability in accurately predicting masked words using the Transformers' `pipeline` function.
