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


## Benchmarks

The **Benchmarks** folder contains files related to the comparison of various machine learning models used in the project.

- **model_benchmarks.ipynb**: 
  A Jupyter Notebook that compares the performance of several machine learning algorithms on the dataset. The models include:
  
  - **Basic Models**: Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).
  - **Advanced Model**: A Neural Network.

  Each model is trained twice: once using the `title` column and once using the `text` column of the dataset. The results are compared to determine the most effective model.

### Key Findings
From the benchmarks, the **SVM model trained with the `title` column** achieved the best results, outperforming the other models.

### Leaderboard Performance
On **November 28, 2024**, using the **SVM model trained with the `title` column**, I ranked:
- **21st out of 44 participants** for SubTask 1 with a score of **0.6899**.
- **10th out of 37 participants** for SubTask 2 with a score of **0.3955**.


## Scores

The **Scores** folder contains the score files generated by the model benchmarks. These scores represent the performance of different machine learning models, as evaluated using various metrics from the benchmark comparisons.


## Images

The **Images** folder contains visual representations comparing the performance of the various models tested in the project.


## Classifiers

The **Classifiers** folder contains Python scripts used to train the models on the cleaned training dataset and make predictions on the validation dataset. Each script trains a different model, either Support Vector Machine (SVM) or Neural Network (NN), using different input features (the `title` or `text` column). After training, the models generate predictions, which are saved in the **Predictions** folder.

### Files in this folder:
- **svm_title_classifier.py**: Trains an SVM model using the `title` column and generates predictions on the validation dataset.
- **svm_text_classifier.py**: Trains an SVM model using the `text` column and generates predictions on the validation dataset.
- **nn_title_classifier.py**: Trains a Neural Network model using the `title` column and generates predictions on the validation dataset.
- **nn_text_classifier.py**: Trains a Neural Network model using the `text` column and generates predictions on the validation dataset.

## Predictions

The **Predictions** folder, as mentioned earlier, contains the prediction files generated by the trained models. These files are in the correct format and ready for submission to the competition. Each file contains the predicted results for the validation dataset, based on the `title` or `text` columns.


## Conclusion

During this project, I explored various machine learning models and techniques for the **Food Hazard Detection Semeval Competition**. While the **SVM model trained on the `title` column** delivered the best results, I also tested a fine-tuned BERT model during the benchmarking process. However, due to its excessive runtime and performance that did not significantly exceed that of simpler models, I decided to exclude it from the final implementation.

The final results demonstrate the effectiveness of simpler models like SVM in this context, making them a practical choice for similar tasks.


## Contact

If you have any questions about this project, feel free to reach out:
- Email: [emm.mavrakis@aueb.gr], Mavrakis Emmanouil.
