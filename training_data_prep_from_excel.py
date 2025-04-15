# -*- coding: utf-8 -*-
"""Data_validation_4th_April_Final_training.ipynb

Original file is located at
    https://colab.research.google.com/drive/1RV2P0Q-hdaMvRQ9q352GW-oQXm5_Rarz
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc


root = "/mnt/win_share1/XELPMOC_2025/"
ub_data_19_m1 = pd.read_parquet(root + '/Artifacts/UB/UB_Cleaned_data_after_manual_verification_19th_March.parquet')
ub_data_19_m2 = pd.read_parquet(root + '/Artifacts/UB/UB_Cleaned_data_after_manual_verification_19th_March_v3.parquet')
ub_data_19_m2_updated = pd.read_parquet(root + '/Artifacts/UB/UB_Cleaned_data_after_manual_verification_19th_March_v4.parquet')

print(ub_data_19_m1.shape, ub_data_19_m2.shape, ub_data_19_m2_updated.shape)
gc.collect()

def compare_dataframes(df1, df2):
  """
  Compares two DataFrames based on 'Image_Name' and creates a new DataFrame with comparison results.

  Args:
      df1: The first DataFrame.
      df2: The second DataFrame.

  Returns:
      A new DataFrame with 'Image_Name' and columns indicating if values are equal for each column.
  """

  # Merge DataFrames on 'Image_Name'
  merged_df = pd.merge(df1, df2, on='Image_Name', how='inner', suffixes=('_df1', '_df2'))

  # Create a new DataFrame to store comparison results
  comparison_df = merged_df[['Image_Name']]

  # Iterate through columns and compare values
  for column in df1.columns:
    if column != 'Image_Name' and column in df2.columns:
       comparison_df[column + '_equal'] = merged_df[column + '_df1'].apply(lambda x: x.strip() if isinstance(x, str) else x) == merged_df[column + '_df2'].apply(lambda x: x.strip() if isinstance(x, str) else x)

  return comparison_df

comparison_result_df = compare_dataframes(ub_data_19_m2, ub_data_19_m2_updated)

def compute_true_false_counts(df):
  """
  Computes the counts of True and False values for each column in a DataFrame.

  Args:
    df: The input DataFrame.

  Returns:
    A new DataFrame with columns: 'column_name', 'True', 'False'.
  """

  result_df = pd.DataFrame(columns=['column_name', 'True', 'False'])

  for column in df.columns:
    # Convert the column to boolean type to ensure proper summation
    true_count = df[column].astype(bool).sum()
    false_count = df[column].size - true_count
    result_df = pd.concat([result_df, pd.DataFrame({'column_name': [column], 'True': [true_count], 'False': [false_count]})], ignore_index=True)

  return result_df

# Example usage (assuming comparison_result_df is your DataFrame from the previous code)
true_false_counts_df = compute_true_false_counts(comparison_result_df)
true_false_counts_df

# Create the output directory if it doesn't exist
output_dir = "/mnt/win_share1/XELPMOC_2025/Processing_stage_2(Img_Json)/UB/06_Apr/JSON"  # Specify your desired output directory
os.makedirs(output_dir, exist_ok=True)

for index, row in tqdm(ub_data_19_m2_updated.iterrows()):
    image_name = row['Image_Name']
    base_name = os.path.splitext(image_name)[0]  # Extract base name without extension
    json_data = row.to_dict()
    del json_data['Image_Name']  # remove Image_Name from JSON data
    output_file = os.path.join(output_dir, f"{base_name}.json")

    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=4)

#Zip all the JSON files into the same folder

import zipfile
import os

def zip_json_files(directory):
  """Zips all JSON files in the specified directory into a single zip file.

  Args:
    directory: The path to the directory containing the JSON files.
  """

  zip_filename = os.path.join(directory, "json_files.zip")  # Name of the zip file
  with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(directory):
      for file in files:
        if file.endswith(".json"):
          file_path = os.path.join(root, file)
          zipf.write(file_path, arcname=os.path.relpath(file_path, directory))

# Example usage
output_dir = "/mnt/win_share1/XELPMOC_2025/Processing_stage_2(Img_Json)/UB/06_Apr/JSON"
zip_json_files(output_dir)