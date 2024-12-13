# Emotion Detection from Text

## Project Overview

**Moodify: Emotion Analysis through Deep Learning**

This innovative project uses the latest machine learning techniques to detect and classify emotions from text data. By leveraging neural network architectures such as LSTM and BiLSTM, we have developed a robust solution for understanding emotions in text.

## Key Objectives

- Develop an intelligent emotion classification system
- Analyze text data using state-of-the-art deep learning models
- Provide insights into emotional context of text

## Dataset Composition

Our dataset comprises Twitter messages meticulously labeled with five primary emotional categories:

| Emotion | Description |
|---------|-------------|
| 😢 Sadness | Expressions of grief, sorrow, or melancholy |
| 😄 Happiness | Texts conveying joy, pleasure, or excitement |
| 😠 Anger | Messages expressing frustration or irritation |
| 😰 Worry | Texts indicating anxiety or concern |
| 🎉 Enthusiasm | Energetic and passionate communications |

**Dataset Source:** [Emotion Classification Dataset](https://github.com/Moodify-Project/Machine-Learning/tree/main/Dataset)

## Key Features

- Advanced emotion classification using deep learning
- Comprehensive text preprocessing and cleaning
- Balanced dataset through intelligent downsampling
- Detailed visualizations of word frequencies
- Flexible and trainable neural network models

## Technology Stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

### Required Libraries

- TensorFlow
- Matplotlib
- NumPy
- Seaborn
- NLTK
- WordCloud
- Scikit-learn

## Model Architectures

### 1. LSTM Model
- **Embedding Layer**: Transforms text into dense vector representations
- **Dropout Layer**: Prevents overfitting
- **LSTM Layer**: Captures sequential dependencies
- **Dense Layer**: Emotion classification with softmax activation

### 2. BiLSTM Model
- **Embedding Layer**: Text vectorization
- **Spatial Dropout**: Advanced regularization
- **Bi-directional LSTM**: Captures context from both directions
- **Dense Layer**: Multi-class emotion prediction

## Performance Metrics

| Model | Training Accuracy | Validation Accuracy |
|-------|------------------|---------------------|
| LSTM | 92.12% | 89.99% |
| BiLSTM | 93.67% | 90.68% |

## Model Visualizations

- [LSTM Architecture](https://github.com/Moodify-Project/Machine-Learning/blob/main/Media/LSTM.png)
- [BiLSTM Architecture](https://github.com/Moodify-Project/Machine-Learning/blob/main/Media/BiLSTM.png)

## Future Improvements

- Expand emotion categories
- Implement transfer learning
