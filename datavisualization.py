import pandas as pd
import os
import boto3
import zipfile
from io import BytesIO
import pydicom
from PIL import Image
import random
import shutil
from data_extraction import extract_data_and_read_csv
extract_data_and_read_csv()
# Define paths
images_root_path = os.path.join(os.getcwd(), "extracted_files/train_images")
output_root_path = "./output_dir"

def read_csv_from_s3(csv_filename):
    s3 = boto3.client('s3')
    bucket_name = 'deeplearning-mlops-demo'
    file_key = 'rsna-2024-lumbar-spine-degenerative-classification.zip'
    with BytesIO() as zip_buffer:
        s3.download_fileobj(bucket_name, file_key, zip_buffer)
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:    
            with zip_ref.open(csv_filename) as csv_file:
                csv_data = pd.read_csv(csv_file)
                print("CSV file read successfully.")
                print(csv_data.head())
    return csv_data
                
csv_filename = 'train_label_coordinates.csv' 
      
df = read_csv_from_s3(csv_filename)
print("======================================", df)

# Function to create directories if they do not exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_dicom_to_jpg(dicom_path, jpg_path):
    dicom_image = pydicom.dcmread(dicom_path)
    image = dicom_image.pixel_array
    im = Image.fromarray(image)
    im = im.convert("L")  # Convert to grayscale
    im.save(jpg_path)

# Iterate over each row in the dataframe
for _, row in df.iterrows():
    study_id = row['study_id']
    series_id = row['series_id']
    instance_number = row['instance_number']
    condition = row['condition']
    
    # Define the source file path
    source_file = os.path.join(images_root_path, str(study_id), str(series_id), f"{instance_number}.dcm")
    
    # Define the destination folder path
    dest_folder = os.path.join(output_root_path, str(condition))
    
    # Create the destination directory if it doesn't exist
    create_dir(dest_folder)
    
    # Define the destination file path
    jpg_file = f"{study_id}_{series_id}_{instance_number}.jpg"
    jpg_path = os.path.join(dest_folder, jpg_file)
    
    # Convert and save the JPG file
    if os.path.exists(source_file):
        convert_dicom_to_jpg(source_file, jpg_path)
        print(f"Saved {jpg_path}")
    else:
        print(f"File {source_file} does not exist")

# Save random images for visualization
num_images_to_select = 1 

for condition_folder in os.listdir(output_root_path):
    condition_path = os.path.join(output_root_path, condition_folder)
    if os.path.isdir(condition_path):
        jpg_files = [f for f in os.listdir(condition_path) if f.endswith('.jpg')]
        random_files = random.sample(jpg_files, min(1, len(jpg_files)))  # Select up to 5 random files
        
        for jpg_file in random_files:
            jpg_path = os.path.join(condition_path, jpg_file)
            new_jpg_path = os.path.join(os.getcwd(), f"{condition_folder}_{jpg_file}")
            
            # Copy the JPG file to CWD
            shutil.copy2(jpg_path, new_jpg_path)
            print(f"Copied {jpg_path} to {new_jpg_path}")

print("Script finished successfully.")
