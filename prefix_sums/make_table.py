""" make_table.py
    for generating pivot tables
    April 2021
"""

import argparse
import glob
import json

import pandas as pd

pd.set_option("display.max_rows", None)
parser = argparse.ArgumentParser(description="Analysis parser")
parser.add_argument("filepath", type=str)
args = parser.parse_args()

df = pd.DataFrame()
for filepath in glob.glob(f"{args.filepath}/*.json"):
    with open(filepath, 'r') as fp:
        data = json.load(fp)

    for d in data.values():
        if not isinstance(d, int):
            if not isinstance(d["test_iter"], int):
                d["test_iter"] = 0

    num_entries = data.pop("num entries")
    little_df = pd.DataFrame.from_dict(data, orient="index")
    df = df.append(little_df)

df["count"] = 1

print(df.keys())
values = ["train_acc", "test_acc", "eval_acc"]
index = ["model", "num_params", "test_mode", "test_iter"]

table = pd.pivot_table(df, index=index, aggfunc={"train_acc": ["min", "max", "mean", "sem"],
                                                 "eval_acc": ["min", "max", "mean", "sem"],
                                                 "count": "count",
                                                 })
print(table)
