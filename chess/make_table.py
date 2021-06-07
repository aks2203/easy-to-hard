""" make_table.py
    for generating pivot tables
    April 2021
"""

import argparse
import json

import pandas as pd

pd.set_option("display.max_rows", None)
parser = argparse.ArgumentParser(description="Analysis parser")
parser.add_argument("filepath", type=str)
args = parser.parse_args()

with open(args.filepath, 'r') as fp:
    data = json.load(fp)

num_entries = data.pop("num entries")
df = pd.DataFrame.from_dict(data, orient="index")
df["count"] = 1

values = ["eval_acc", "train_acc", "test_acc"]
index = ["model", "test_iter", "eval_start", "eval_end"]

table = pd.pivot_table(df, index=index, aggfunc={"eval_acc": ["mean", "sem"],
                                                 "train_acc": ["mean", "sem"],
                                                 "test_acc": ["mean", "sem"],
                                                 "count": "count",
                                                 })
print(table)
