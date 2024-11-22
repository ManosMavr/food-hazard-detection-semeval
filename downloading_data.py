import os
import zipfile
import requests

# Create a folder named "datasets" if it doesn't exist
os.makedirs('datasets', exist_ok=True)

# Helper function to save files to the "datasets" folder
def save_to_datasets(filename, content):
    filepath = os.path.join('datasets', filename)
    with open(filepath, 'wb') as file:
        file.write(content)
    return filepath

# Download training data (labeled)
url = 'https://raw.githubusercontent.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/refs/heads/main/data/incidents_train.csv'
response = requests.get(url)
train_file = save_to_datasets('incidents_train.csv', response.content)

# Download and unzip the validation file
url = 'https://codalab.lisn.upsaclay.fr/my/datasets/download/26c12bc0-3878-4edf-8b4a-9682763c0b7e'
response = requests.get(url)
val_zip = save_to_datasets('dataset.zip', response.content)

with zipfile.ZipFile(val_zip, 'r') as zip_ref:
    zip_ref.extractall('datasets')  # Extract directly to the "datasets" folder
    extracted_files = zip_ref.namelist()
    for extracted_file in extracted_files:
        if extracted_file.lower().endswith('.csv'):
            extracted_path = os.path.join('datasets', extracted_file)
            renamed_path = os.path.join('datasets', 'incidents_val.csv')
            os.rename(extracted_path, renamed_path)
            print(f"Renamed {extracted_file} to incidents_val.csv")

# Clean up any leftover "incidents" file
incidents_path = os.path.join('datasets', 'incidents')
if os.path.exists(incidents_path):
    os.remove(incidents_path)
    print("Deleted leftover file 'incidents'")

# Remove the validation ZIP file
if os.path.exists(val_zip):
    os.remove(val_zip)
    print("Removed dataset.zip")


# Download and unzip the test file
url = 'https://codalab.lisn.upsaclay.fr/my/datasets/download/5695a2da-4c2b-4447-8c0c-2a30252f648c'
response = requests.get(url)
test_zip = save_to_datasets('public_dat.zip', response.content)

with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall('datasets')  # Extract directly to the "datasets" folder
    extracted_files = zip_ref.namelist()
    for extracted_file in extracted_files:
        if extracted_file.lower().endswith('.csv'):
            extracted_path = os.path.join('datasets', extracted_file)
            renamed_path = os.path.join('datasets', 'incidents_test.csv')
            os.rename(extracted_path, renamed_path)
            print(f"Renamed {extracted_file} to incidents_test.csv")


# Remove the test ZIP file
if os.path.exists(test_zip):
    os.remove(test_zip)
    print("Removed public_dat.zip")

print("All files processed and stored in the 'datasets' folder.")