from sklearn.model_selection import train_test_split
import zipfile
import os
import requests

# Download training data (labeled):
url = 'https://raw.githubusercontent.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/refs/heads/main/data/incidents_train.csv'
response = requests.get(url)

with open('incidents_train.csv', 'wb') as file:
    file.write(response.content)




#Download the validation file
url = 'https://codalab.lisn.upsaclay.fr/my/datasets/download/26c12bc0-3878-4edf-8b4a-9682763c0b7e'
response = requests.get(url)

#Save the content
with open('dataset.zip', 'wb') as file:
    file.write(response.content)

#Unzip the downloaded file
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()  # You can specify the folder where you want to extract
    
# List the files in the zip and assume there's only one CSV file
    extracted_file = zip_ref.namelist()[0]  # Get the first (and only) file
    if extracted_file.lower().endswith('.csv'):  # Ensure it's a CSV file
        # Rename the file
        os.rename(extracted_file, 'incidents_val.csv')
        print(f"Renamed {extracted_file} to incidents_val.csv")


#Download the test file
url = 'https://codalab.lisn.upsaclay.fr/my/datasets/download/5695a2da-4c2b-4447-8c0c-2a30252f648c'
response = requests.get(url)

#Save the content
with open('public_dat.zip', 'wb') as file:
    file.write(response.content)

#Unzip the downloaded file
with zipfile.ZipFile('public_dat.zip', 'r') as zip_ref:
    zip_ref.extractall()  # You can specify the folder where you want to extract
    
# List the files in the zip and assume there's only one CSV file
    extracted_file = zip_ref.namelist()[0]  # Get the first (and only) file
    if extracted_file.lower().endswith('.csv'):  # Ensure it's a CSV file
        # Rename the file
        os.rename(extracted_file, 'incidents_test.csv')
        print(f"Renamed {extracted_file} to incidents_test.csv")