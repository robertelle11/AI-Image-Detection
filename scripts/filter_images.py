import os
import pandas as pd

folder_path = r"C:\Users\mgr61\Downloads\africa_images\Streetview_Image_Dataset"
input_csv = r"C:\Users\mgr61\OneDrive\Documents\africa_coordinates.csv"
output_csv = r"C:\Users\mgr61\OneDrive\Documents\africa_images.csv"
column = "File Path"
start = 50000
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
