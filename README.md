# Food Hazard Detection for Semeval Competition

## Introduction

This project was developed as part of the **Practical Data Science** course in the **MSc in Data Science** program at **AUEB** (Athens University of Economics and Business). The assignment involved participating in the **Food Hazard Detection** task from the **Semeval Competition**.

The goal of this project is to train machine learning models to predict food-related hazards based on text data. The models are designed to automatically detect food safety issues by analyzing text data from various online sources such as social media, forums, and other web platforms.

In this project, different machine learning models, including **Neural Networks** (NN) and **Support Vector Machines** (SVM), were trained and evaluated for their performance in predicting food hazards, with the ultimate aim of achieving high prediction scores in the competition.


## Requirements

To run this project, you'll need to have the following libraries installed.

- `matplotlib==3.4.3`: For data visualization and plotting.
- `nltk==3.6.3`: For Natural Language Processing tasks.
- `numpy==1.21.2`: For numerical operations and handling arrays.
- `pandas==1.3.3`: For data manipulation and analysis.
- `scikit-learn==0.24.2`: For machine learning algorithms and model evaluation.
- `seaborn==0.11.1`: For enhanced visualization of data.
- `torch==1.9.0`: For building and training Neural Networks (PyTorch framework).

You can install all dependencies by running the following command:

```bash
pip install -r requirements.txt
```


## Data Processing

The **Data Processing** folder contains the script used for downloading and preparing the datasets and the notebook for the data processing. Below are the files in this folder and their purposes:

- **downloading_data.py**: 
  A Python script that downloads the training and validation datasets provided by the competition and saves them in the `Datasets` folder. The script uses the competition's links to fetch the data.

- **data_cleaning.ipynb**: 
  A Jupyter Notebook that describes the data cleaning process. It performs cleaning operations on the raw datasets and saves the processed, clean CSV files in the `Datasets` folder for further use.

## Datasets

The **Dataset** folder contains all the datasets used in the project, including both the initial datasets provided by the competition and the cleaned datasets generated during the data processing phase.

### Raw Datasets
These datasets were provided by the competition for training the models and participating in the competition:
- **incidents_train.csv**: The raw training dataset.
- **incidents_test.csv**: The raw testing dataset.
- **incidents_val.csv**: The raw validation dataset.

### Cleaned Datasets
These datasets are the cleaned versions of the raw datasets, prepared for model training and validation:
- **cleaned_train.csv**: The cleaned training dataset.
- **cleaned_test.csv**: The cleaned testing dataset.
- **cleaned_val.csv**: The cleaned validation dataset.


