"""Aggregate classification results from json files in nice table."""

#%%
import os
import glob
import pandas as pd
import json


def json_load(filepath):
    with open(filepath, 'r') as json_file:
        return json.load(json_file)


seed_dir = 'seed_7899463'

DATA_DIR = os.path.join('data/classification_results',
                        seed_dir)
DF_PATH = os.path.join('experiments/manuscript/',
                       'classification_results.csv')
dicts = [
    json_load(filepath)
    for filepath in glob.glob(os.path.join(DATA_DIR, '*.json'))
]

# parametric cross validation results
# csvs = [
#     pd.read_csv(filepath)
#     for filepath in glob.glob(os.path.join(DATA_DIR, '*.csv'))
# ]

df = pd.io.json.json_normalize(dicts)
df.columns = [
    c.replace('.', '_').replace('validation_scores_', '') for c in df.columns]

df.to_csv(DF_PATH)

#%%
