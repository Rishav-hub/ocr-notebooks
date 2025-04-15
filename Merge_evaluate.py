# -*- coding: utf-8 -*-
"""UB_Testset_Merge_20th_March.ipynb

Original file is located at
    https://colab.research.google.com/drive/1_QF-XKhozj1rrdflfoG6hTkY-wegkLPD
"""

import pandas as pd
import gc
from tqdm import tqdm
gc.collect()

ub_overal_data= pd.read_parquet("./UB_testset_222_files_groundtruth_from_parse_files.parquet")
donut_output_test_data = pd.read_parquet("./UB_validation_results_UB_20_March_model_231_images.parquet")

donut_output_test_data.rename(columns={'Key':'Field_Name', 'Value':'Output',
                                       'Image_name': "Image_Name"}, inplace=True)


print(ub_overal_data.shape, donut_output_test_data.shape)
print(donut_output_test_data['Field_Name'].nunique())

#Filtering the required fields
ub_overal_data = ub_overal_data[ub_overal_data['Field_Name'].isin(donut_output_test_data['Field_Name'].unique())]
print(ub_overal_data.shape, donut_output_test_data.shape)

#Checking the number of images infered and GT extracted
ub_overal_data['Image_Name'].nunique(), donut_output_test_data['Image_Name'].nunique()

# ub_overal_data = ub_overal_data[ub_overal_data['Page'] == '1']
# ub_overal_data.shape

inference_images = donut_output_test_data['Image_Name'].unique()
ground_truth_images = ub_overal_data['Image_Name'].unique()

missing_images = list(set(ground_truth_images) - set(inference_images))
print(len(missing_images))

missing_images_2 = list(set(inference_images) -set(ground_truth_images))
print(len(missing_images_2))

ub_overal_data = ub_overal_data[~ub_overal_data["Image_Name"].isin(missing_images)]
donut_output_test_data = donut_output_test_data[~donut_output_test_data["Image_Name"].isin(missing_images_2)]

print(ub_overal_data.shape, donut_output_test_data.shape)

donut_output_test_data[donut_output_test_data['Field_Name'] == '9_PatAddr1']

print(ub_overal_data["Image_Name"].nunique(), donut_output_test_data["Image_Name"].nunique())

gc.collect()
tqdm.pandas()

merged_data_frame_all_images= pd.DataFrame()
for img_name, grp_b_im in tqdm(ub_overal_data.groupby('Image_Name')):
#     grp_b_im = grp_b_im.sort_values(by=['y1'])
    counts = grp_b_im['Field_Name'].value_counts()
    mask = grp_b_im['Field_Name'].map(counts) > 1
    grp_b_im.loc[mask, 'Field_Name'] = grp_b_im['Field_Name'] + (grp_b_im.groupby('Field_Name').cumcount() + 1).astype(str)

    donut_output_for_img = donut_output_test_data[donut_output_test_data["Image_Name"] == img_name]
    counts = donut_output_for_img['Field_Name'].value_counts()
    mask = donut_output_for_img['Field_Name'].map(counts) > 1
    donut_output_for_img.loc[mask, 'Field_Name'] = donut_output_for_img['Field_Name'] + (donut_output_for_img.groupby('Field_Name').cumcount() + 1).astype(str)
    each_image_merged = pd.merge(grp_b_im, donut_output_for_img , on=['Image_Name','Field_Name'], how='outer')
    merged_data_frame_all_images = pd.concat([merged_data_frame_all_images, each_image_merged])
    merged_data_frame_all_images.fillna("[BLANK]")
    # del each_image_merged
    # gc.collect()

merged_data_frame_all_images.fillna("[BLANK]", inplace=True)
merged_data_frame_all_images["XELP_GT"] = merged_data_frame_all_images.progress_apply(lambda x: 1 if (str(x["Output"]).strip() == str(x["Data"]).strip()) else 0, axis=1)

print("\n")
print(merged_data_frame_all_images["XELP_GT"].mean())

import re
def normalize_field_name(field_name):
  """Normalizes the field name by removing the last digits, except for fields ending with Addr1."""
  return re.sub(r'\d+$', '', field_name)

merged_data_frame_all_images['Normalized_Field_Name'] = merged_data_frame_all_images['Field_Name'].apply(normalize_field_name)
print(merged_data_frame_all_images.shape)

xel_gt_percentage = merged_data_frame_all_images.groupby('Normalized_Field_Name')['XELP_GT'].mean().sort_values(ascending=True)
print(xel_gt_percentage[:60])

merged_data_frame_all_images[(merged_data_frame_all_images["XELP_GT"] == 0) & (merged_data_frame_all_images['Normalized_Field_Name'] == '1_BillProvOrgName')][['Image_Name', 'Field_Name','Data', 'Output']][:60]
merged_data_frame_all_images.head()
# merged_data_frame_all_images.to_csv("UB_204_merged_data_frame_all_images.csv")