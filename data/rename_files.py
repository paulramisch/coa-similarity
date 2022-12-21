# Lightroom changes the export names if there are certain characters (e.g. ".") in the name
# This script creates a copy of the input folder with the inputs renamend to match the Lightroom export

import os
import pandas as pd
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Parameter
original_path = "../data/coa"
copy_path = "../data/coa_rotated_"
renamed_path = "../data/coa_rotated"
shortened_len = 9


# Function to get images from folder
def get_content(directory):
    all_imgs = []
    for file in os.listdir(directory):
        # check only text files
        if file.lower().endswith(('.png', 'jpg')):
            all_imgs.append(file)
    return all_imgs


# Parameter
original = get_content(original_path)
copy = get_content(copy_path)

# List of shortend namens of files that are missing in the renamend copy
df_og_short = []

for file in original:
    if file not in copy:
        df_temp = pd.DataFrame([[file, file[0:shortened_len], True]], columns=["title", "title_short", "changed"])
        df_og_short.append(df_temp)
    else:
        df_temp = pd.DataFrame([[file, file[0:shortened_len], False]], columns=["title", "title_short", "changed"])
        df_og_short.append(df_temp)

df_og_short = pd.concat(df_og_short)

duplicateList = []
# Check there are no duplicates
if len(df_og_short[(df_og_short.changed) & (df_og_short["title_short"].duplicated())]) > 0:
    og_renamed_short = df_og_short[(df_og_short.changed) & (df_og_short["title_short"].duplicated())].title_short
    uniqueList = []

    for i in og_renamed_short:
        if i not in uniqueList:
            uniqueList.append(i)
        elif i not in duplicateList:
            duplicateList.append(i)

    print("Attention! There are duplicates!:", len(duplicateList))

# Create empy list of files that need manual copying due to possible name conflicts
manual_copy_needed = []

# Create empty list of copied/missing files
copied_og = []
missing_og = []

# Iterate over copies
for file in copy:
    if len(df_og_short[df_og_short.title == file]) > 0:
        shutil.copyfile(original_path + "/" + file, renamed_path + "/" + file)
        copied_og.append(file[0:shortened_len])
        t = 0
    # Check if the file name was changed
    elif (df_og_short[df_og_short.changed == True]['title_short'].eq(file[0:shortened_len])).any():
        # If the file name was changed and no duplicate name, copy the file
        if file[0:shortened_len] not in duplicateList:
            file_to_copied = df_og_short[(df_og_short.changed == True)
                                         & (df_og_short.title_short == file[0:shortened_len])].title[0]
            shutil.copyfile(original_path + "/" + file_to_copied, renamed_path + "/" + file)
            copied_og.append(df_og_short[(df_og_short.changed == True)
                                         & (df_og_short.title_short == file[0:shortened_len])].title[0])
        else:
            # Names that are not in the shortened list
            manual_copy_needed.append(file)
            print('Manual copy needed:', file)
    # If the name wasn't changed, just copy it
    else:
        print("Check needed:", file)

# Check for missing/non-matched img
for file in original:
    if file[0:shortened_len] not in copied_og:
        missing_og.append(file)

print(f"Number of missing original images: {len(missing_og)}")