# Emotion Detection from text
Designing a deep learning model to detect emotions from text using TensorFlow, Developing LSTM and BiLSTM models to analyze and classify emotions based on sentences provided by users.

## Project Overview
This project aims to build a machine learning model that detects emotions from text, using a dataset containing text labeled with emotions such as 'sadness', 'happiness', 'anger', 'worry', and 'enthusiasm'. After data cleaning and downsampling for class balance, LSTM and BiLSTM models were trained to predict emotions. This model will be implemented in applications "Moodify" to analyze user emotions based on their text, using Python, TensorFlow, and Keras.


## Dataset
The dataset consists of Twitter messages labeled with emotions. Each entry in the dataset represents a text segment and the dominant emotion expressed. The emotions are classified into five categories:
- Sadness
- Happiness
- Anger
- Worry
- Enthusiasm
  
The dataset is available [here](https://github.com/Moodify-Project/Machine-Learning/tree/main/Dataset)

## Feature
- Emotion classification from text data
- Cleaned and preprocessed text using NLP techniques
- Balanced dataset using downsampling for each class
- Visualizations of word frequencies and word clouds for each emotion
- Trainable deep learning models (LSTM, BiLSTM) for emotion classification

## Requirements 
- Tensorflow 
- Matplothlib
- Numpy
- Seaborn
- NLTK
- WordCloud
- Scikit-learn
- NFX

## Model Architecture
Two models were developed and trained to classify emotions from text:
LSTM Model:
- Embedding layer
- Dropout layer for regularization
- LSTM layer for sequential data processing
- Dense layer with softmax activation for emotion classification

BiLSTM Model:
- Embedding layer
- Spatial Dropout layer for regularization
- Bi-directional LSTM layers for better sequential data handling
- Dense layer with softmax activation for emotion classification

Both models were compiled with the Adam optimizer and categorical cross-entropy loss function.

## Documentation
![Deskripsi Gambar](https://github.com/Moodify-Project/Machine-Learning/blob/main/Media/LSTM.png)
![Deskripsi Gambar](https://github.com/Moodify-Project/Machine-Learning/blob/main/Media/BiLSTM.png)

## Results
LSTM Model: The model shows steady improvement, achieving a final training accuracy of 92.12% with a validation accuracy of 89.99%.
BiLSTM Model: The BiLSTM model outperformed the LSTM model with a final training accuracy of 93.67% and a validation accuracy of 90.68% after.



