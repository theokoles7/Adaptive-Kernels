from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


labels = pd.read_csv("/content/drive/MyDrive/COLAB/Data_Colab/Medical_Classification/Pneumonia_Project/Label/stage_2_train_labels.csv")
print(labels.head(6))

# Remove duplicate entries
labels = labels.drop_duplicates("patientId")
print(labels.head(6))
READ_PATH = Path("/content/drive/MyDrive/COLAB/Data_Colab/Medical_Classification/Pneumonia_Project/RawData/stage_2_train_images/")
SAVE_PATH = Path("/content/drive/MyDrive/COLAB/Data_Colab/Medical_Classification/Pneumonia_Project/Processed_Data/")

sums = 0
sums_squared = 0

for c, patient_id in enumerate(tqdm(labels.patientId)):
  dcm_path = READ_PATH / patient_id  # Create the path to the dcm file
  dcm_path = dcm_path.with_suffix(".dcm")  # And add the .dcm suffix

  # Read the dicom file with pydicom and standardize the array
  dcm = pydicom.read_file(dcm_path).pixel_array / 255

  # Resize the image as 1024x1024 is way to large to be handeled by Deep Learning models at the moment
  # Let's use a shape of 224x224
  # In order to use less space when storing the image we convert it to float16
  dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)

  # Retrieve the corresponding label
  label = labels.Target.iloc[c]

  # 4/5 train split, 1/5 val split
  if c < 24000:
    train_or_val = "train"
  else:
    train_or_val = "val"

  current_save_path = SAVE_PATH / train_or_val / str(label)  # Define save path and create if necessary
  current_save_path.mkdir(parents=True, exist_ok=True)
  np.save(current_save_path / patient_id, dcm_array)  # Save the array in the corresponding directory
  # print(str(c) + ": Directory: " + str(current_save_path))

  normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image
  if train_or_val == "train":  # Only use train data to compute dataset statistics
    sums += np.sum(dcm_array) / normalizer
    sums_squared += (np.power(dcm_array, 2).sum()) / normalizer