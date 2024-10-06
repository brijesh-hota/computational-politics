# Computational Politics: Advancing Sentiment Analysis through Natural Language Processing in Election Studies

## Members:
- Mundhir Al Bohri (B00858907) - mn409130@dal.ca
- Harsahib Preet Singh (B00850322) - hr6446@dal.ca
- Brijesh Hota (B00844617) - br747252@dal.ca

## Problem Statement:
The Canadian federal elections are organized every 4 years where different candidates are elected amongst the various political parties to secure a party seat in Canada’s House of Common where they represent their riding as Member of Parliament (MP). Social media (like Twitter, Reddit, etc.) provide voters an open platform to convey different opinions about competing candidates and political parties. Understanding the sentiments of the voters is as important to other voters in the electorate as it is for different contesting political parties. Natural Language Processing (NLP) techniques can help us to effectively decipher sentiments expressed by the electorate during election campaigns using data from various social media platforms.

## List of Possible Approaches:
### i. Probabilistic Models (Naive Bayes Classifier):
Bayes' theorem is a fundamental model for sentiment analysis, known for its simplicity, efficiency, and effectiveness in producing reliable results. We could say it is the Swiss Army Knife of machine learning algorithms, it is versatile and particularly adept at handling small to medium-sized datasets. For our purposes, we will employ the Naive Bayes model as a foundational layer for sentiment analysis. However, it is important to note that our utilization of this model will be confined to basic sentiment analysis and simple analytical tasks.
Bayes' theorem can be expressed in two forms:
1) First form 𝑃(𝐴|𝐵) = 𝑃(𝐵|𝐴) ⋅ 𝑃(𝐴) / 𝑃(𝐵)
2) The second form is based on breaking the set 𝐵 into disjoint sets 𝐵 = ⋃ 𝐴𝑖 𝑃(𝐴𝑖|𝐵) = 𝑃(𝐵|𝐴𝑖) ⋅ 𝑃(𝐴𝑖) / ∑ (𝐵|𝐴𝑖)𝑃(𝐴𝑖)

The prediction formula for our model, where 𝐶 represents the probability of being political or non-political and 𝑊 represents the words, is shown below:
𝑐𝑜𝑛𝑐𝑙𝑢𝑠𝑖𝑜𝑛 = arg max 𝐶 𝑃(𝐶|𝑊) = arg max 𝐶 𝑃(𝑊|𝐶) ⋅ 𝑃(𝐶) / 𝑃(𝑊) = arg max 𝐶 𝑃(𝑊|𝐶) ⋅ 𝑃(𝐶) where 𝑃(𝑊 | 𝐶) = ∏ 𝑃(𝑤𝑖 | 𝐶)𝑛

### ii. Transformer Models:
Transformer models, introduced in the paper "Attention is All You Need", have revolutionized the field of natural language processing. They have laid the groundwork for large language models such as BERT and GPT. In contrast to past neural network models like LSTMs (Long Short-Term Memory Networks) and RNNs (Recurrent Neural Networks), which were constrained by sequential data processing and required significant computing power, transformers are much more efficient. The main strength of transformers lies in their ability to process text in parallel, significantly reducing training time while delivering superior performance. This allows capturing dependencies in long texts with which traditional neural networks struggle.

A transformer model is composed of two main parts: an encoder and a decoder. Each consists of a stack of layers that process the input sequence in parallel, as opposed to sequentially. The pivotal components of a transformer model include the self-attention mechanism, multi-head attention, and positional encoding.

1. Self-Attention Mechanism
The self-attention mechanism allows the model to weigh the importance of each word in the input sequence based on its relationship with other words.

2. Multi-Head Attention
Multi-head attention consists of multiple self-attention layers running in parallel, each with its own set of parameters. The output of each self-attention layer is concatenated and linearly transformed to produce the final output.

3. Positional Encoding
Since the transformer model processes the input sequence in parallel, it does not have any sense of order. To address this, positional encoding is added to the input embeddings to give the model information about the position of each word in the sequence.

Transformer models have demonstrated significant effectiveness across a variety of NLP tasks. Presently, numerous pre-trained models are freely accessible. Utilizing these models is a compelling approach for political analysis, as it necessitates a comprehensive foundational knowledge to identify names, phrases, and the behavior of individuals behind the tweets.

## Project Plan for Future:
- Data pre-processing: We plan to kick-off the project with the dataset pre-processing tasks. We have finalized our datasets which will be currently open datasets from Kaggle covering the Tweets of 44th Canadian General Elections. The data columns will be analysed for the different correlation amongst them. We will also plot different Histograms, Bar plots and Stacker'd charts to understand the dataset better.
- Model Selection: As we finish pre-processing our dataset, we begin working on the NLP model. We would construct a basic model using Naïve Bayes approach for basic sentiment analysis. As we get a basic idea of the model’s performance, we would move forward to develop a Transformer based model using Google’s BERT-base model. We would be using Hugging Face model repository to import our base model and fine-tune it to our needs based on the initial Naïve Bayes model.
- Model Evaluation: We’d understand our model’s performance based on different evaluation metrics like F1 score, accuracy, and precision. As we finish analysing our model, we’d fine-tune our Transformer model’s hyperparameters and plot different curves to showcase if the model is overfitting or underfitting.

## References:
- [1] V. Keselj, "Introduction to Probabilistic NLP Notes," in CSCI 4152/6509 - Natural Language Processing, Faculty of Computer Science, Dalhousie University, Halifax, NS.
- [2] A. Vaswani et al., “Attention is all you need,” arXiv.org, https://arxiv.org/abs/1706.03762.

## Possible References for Future Work:
- [1] Mr. V. Chandra Sekhar Reddy, K. Manvith Reddy, CH. Vachan Sai, K. Suraj, A. Abhinash, "A Survey on Automated Sentimental Analysis of Twitter Data using Supervised Algorithm", International Journal of Advanced Research in Science, Communication and Technology, pp.196, 2022.
- [2] C. Ziems, W. Held, O. Shaikh, J. Chen, Z. Zhang, and D. Yang,
