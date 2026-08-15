# Deep Learning Notes

## Neural Networks

A neural network is a computational model inspired by biological neurons. It consists of layers of interconnected nodes that transform input data through weighted connections and activation functions.

Feedforward neural networks pass information in one direction: input layer → hidden layers → output layer.

## Supervised Machine Learning

Supervised learning uses labeled training data to learn a mapping from inputs to outputs. The two main tasks are classification (predicting categories) and regression (predicting continuous values).

Traditional supervised ML algorithms include:
- Linear regression
- Logistic regression
- Decision trees
- Support vector machines
- Random Forest

## Deep Learning

Deep learning uses neural networks with many hidden layers to learn hierarchical representations of data. Unlike traditional ML, deep learning can automatically discover features from raw data.

Key advantages of deep learning:
1. Automatic feature extraction
2. Strong performance on unstructured data (images, text, audio)
3. Scales well with large datasets and compute

Key disadvantages:
1. Requires large amounts of data
2. Computationally expensive
3. Less interpretable than traditional models

## Comparison: Traditional ML vs Deep Learning

| Aspect | Traditional ML | Deep Learning |
|--------|---------------|---------------|
| Feature engineering | Manual | Automatic |
| Data requirements | Smaller datasets | Large datasets |
| Interpretability | Higher | Lower |
| Compute needs | Moderate | High |
| Best for | Structured/tabular data | Images, text, speech |

## Convolutional Neural Networks

CNNs are specialized for grid-like data such as images. They use convolutional layers to detect local patterns like edges, textures, and shapes.

Applications include image classification, object detection, and medical imaging.

## Transformers

Transformers use self-attention mechanisms to process sequences in parallel. They are the foundation of modern large language models (LLMs).

The attention mechanism allows the model to weigh the importance of different parts of the input when producing output.
