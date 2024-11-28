# Loading Packages
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd



# Neural Network Definition
class TextClassifierNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TextClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def training(vectorizer,column):
    vectorizer.fit(trainset[column])
    X_train = vectorizer.transform(trainset[column]).toarray()
    X_val = vectorizer.transform(val_data[column]).toarray()

    
    # Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 128  # Adjust based on experiments
    batch_size = 32
    epochs = 10
    learning_rate = 1e-3

    # Create label encoders for each target column
    label_encoders = {}

    for label in ('hazard-category', 'product-category', 'hazard', 'product'):
        print(label.upper())

        # Encode labels
        le = LabelEncoder()
        le.fit(trainset[label])
        trainset[label] = le.transform(trainset[label])
        label_encoders[label] = le

        # Prepare target labels
        y_train = trainset[label].values

        # Get number of classes
        num_classes = len(le.classes_)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)


        # **Initialize the model for this label**
        model = TextClassifierNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim, num_classes=num_classes)
        model.to(device)

        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size].to(device)
                batch_y = y_train_tensor[i:i + batch_size].to(device)

                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Predicting the val set
        model.eval()  # **Model is now defined here**
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_predictions = torch.argmax(val_outputs, axis=1).cpu().numpy()

        # Decode predictions back to string labels
        val_data[label] = label_encoders[label].inverse_transform(val_predictions)
        # Print confirmation
        print(f"Predictions for {label} saved to '{label}' column.")


if __name__ == "__main__":
    #Loading the data
    trainset = pd.read_csv('Datasets/cleaned_train.csv', index_col=0)
    val_data = pd.read_csv('Datasets/cleaned_val.csv', index_col=0)

    #Initializing the folder path
    text_nn_pred_path = 'Predictions/text_nn'

    #Create the folder to save the predictions
    os.makedirs(text_nn_pred_path, exist_ok=True)

    #Defining the file path
    text_file_path = os.path.join(text_nn_pred_path, 'submission.csv')

    # TF-IDF Vectorizer for text
    text_tfidf_vect = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,2), max_df=0.5, min_df=5)

    print('Training the Neural Network text model.')
    
    training(text_tfidf_vect,'text')

    #Saving the DataFrame to the csv file
    val_data.to_csv(text_file_path, index=False)