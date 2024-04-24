import glob
import re
import sys

import pandas as pd

# import argparse

# parser = argparse.ArgumentParser(description='Argument parser with a boolean argument')
# parser.add_argument('--folder', type=str)
# args = parser.parse_args()

folder = sys.argv[1]
print(folder)
files = glob.glob(f"{folder}/*out")
print("Number of files:", len(files))


def read_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return data.decode("utf-8")


def extract_final_acc(data):
    pattern = r'final_mean_metric = (\d+\.\d+)'

    match = re.search(pattern, data)

    if match:
        extracted_value = match.group(1)
        return extracted_value
    else:
        return None


data = read_file(files[0])

files_acc = []
for f in files:
    data = read_file(f)
    acc = extract_final_acc(data)
    if acc:
        files_acc.append([f, float(acc)])

df = pd.DataFrame(files_acc, columns=["file_path", "acc"])
print(df)

df["lrs"] = df["file_path"].apply(lambda x: x.split("/")[-1].split(".out")[0].split("-")[1:])
# print(df["lrs"])
df["lr_weight"] = df["lrs"].apply(lambda x: float(x[0]))
df["lr_struc"] = df["lrs"].apply(lambda x: float(x[1]))
df["lr_weight_decay_rate"] = df["lrs"].apply(lambda x: float(x[2]))
df["seed"] = df["lrs"].apply(lambda x: int(x[-1][4:]))

df = df.loc[:, ["file_path", "lr_weight", "lr_struc", "lr_weight_decay_rate", "seed", "acc"]]
df = df.sort_values(by=["acc", "lr_weight", "lr_struc", "lr_weight_decay_rate", "seed"],
                    ascending=False).reset_index(drop=True)
print(df.head(60))
df["acc"] = df["acc"].apply(float)

df_result = df.groupby(by=["lr_weight", "lr_struc", "lr_weight_decay_rate"]).agg(
    mean_mean_acc=("acc", "mean"),
    std_mean_acc=("acc", "std"),
)
print(df_result.head(60))
print(df["file_path"][0])
