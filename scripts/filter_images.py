import os
import pandas as pd
from pathlib import Path

folder_path = path = Path(__file__).parent / r"..\spreadsheets"
input_csv = path = Path(__file__).parent / r"..\random_canadas_images.csv"
output_csv = path = Path(__file__).parent / r"..\canada_images.csv"
column = "File Path"
start = 11600
ext = ".png"

df = pd.read_csv(input_csv)
df.reset_index(drop=True, inplace=True)

needed = set(df[column])
all_files = set(os.listdir(folder_path))
for file in all_files:
    if file not in needed:
        os.remove(os.path.join(folder_path, file))


renamed = []
for i, row in df.iterrows():
    old = row[column]
    new = f"{start + i}{ext}"
    old_path = os.path.join(folder_path, old)
    new_path = os.path.join(folder_path, new)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        renamed.append(new)
    else:
        renamed.append("")

df[column] = renamed
df.to_csv(output_csv, index=False)
