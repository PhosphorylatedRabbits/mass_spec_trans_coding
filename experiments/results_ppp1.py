"""Explore results with various plots."""

#%%
if 'get_ipython' in dir():
    # in case interactive session was moved to __file__ dir
    %cd ..
#%%
import os
import glob
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def json_load(filepath):
    with open(filepath, 'r') as json_file:
        return json.load(json_file)


seed_dir = 'seed_7899463'

# or where downloaded from Box
DATA_DIR = os.path.join('data/classification_results',
                        seed_dir)

dicts = [
    json_load(filepath)
    for filepath in glob.glob(os.path.join(DATA_DIR, '*.json'))
]

# csvs = [
#     pd.read_csv(filepath)
#     for filepath in glob.glob(os.path.join(DATA_DIR, '*.csv'))
# ]

df = pd.io.json.json_normalize(dicts)
df.columns = [c.replace('.', '_') for c in df.columns]

#remove most mobilenets
drop_nets = [
    'mobilenet_v1_100_224',
    'mobilenet_v2_035_224',
    'mobilenet_v2_140_224',
    'mobilenet_v2_100_224',
    'mobilenet_v1_025_128',
    'mobilenet_v1_100_128',
    'mobilenet_v2_100_96',
    'mobilenet_v2_050_128',
    'mobilenet_v2_035_128',
]
df = df[df["module"].apply(lambda x: x not in drop_nets)]


df_ = df.sort_values(by='validation_scores_AUC', ascending=False)[[
    'classifier',
    'validation_scores_AUC',
    'validation_scores_Accuracy',
    'validation_scores_F1',
    # 'cv_index',
    'training_scores_mean_test_AUC',
    'training_scores_mean_test_Accuracy',
    'training_scores_mean_test_F1',
    'training_scores_mean_train_AUC',
    'training_scores_mean_train_Accuracy',
    'training_scores_mean_train_F1',
    'encoded_features_size',
    'non_varying_features',
    # # 'encoded_image_size',
    # 'encoded_image_size_height',
    'encoded_image_size_width',
    'modality',
    'module',
    # # 'raw_image_size',
    # 'raw_image_size_height',
    'raw_image_size_width',
    'cohort_identifier',
    # # only in rerun
    'validation_scores_Brier_Loss',
    'validation_scores_DP',
    'validation_scores_Log_Loss',
    'validation_scores_P+',
    'validation_scores_P-',
    'validation_scores_Precision',
    'validation_scores_Recall',
    'validation_scores_Specificity',
    'validation_scores_Youdens'
]]
order = df_['module'].unique()

# df_[(df_['validation_scores_AUC'] >= 0.75)][
df_[[
    'validation_scores_AUC',
    'validation_scores_Accuracy',
    'modality',
    'module',
    'cohort_identifier',
    'classifier'
]].head(40)

# remove cropped
df_ = df_.query(
    "(raw_image_size_width != '243.0') & (raw_image_size_width != '973.0')"
)
# only all_modalitues
df__ = df_.query("modality == 'all_modalities'")


#%% message: cropped is the same as ms1_only
data = df
g = sns.catplot(x="classifier", y="validation_scores_AUC",
                # hue="module",
                dodge=True,
                row="modality",
                col="raw_image_size_width",
                data=data, kind="box",
                palette='colorblind'
                )

#%% message: all_modalities is better
data = df_
g = sns.catplot(x="modality", y="validation_scores_AUC",
                hue="module",
                dodge=True,
                col="classifier",
                data=data, kind="strip",
                # jitter=True
                )

#%% pairwise t-test ms1_only vs all_modalities
modalities_paired = df_.query("cohort_identifier != 'proteomics'").set_index(
    ['classifier', 'module', 'cohort_identifier']
).pivot(columns='modality')
statistic, p_value = stats.ttest_rel(
    modalities_paired.loc[:, ('validation_scores_AUC', 'ms1_only')].values,
    modalities_paired.loc[:, ('validation_scores_AUC', 'all_modalities')].values,  # noqa
)

data = (modalities_paired.xs(
    'ms1_only', level='modality', axis=1
) - modalities_paired.xs(
    'all_modalities', level='modality', axis=1
)).reset_index()
p_value, data['validation_scores_AUC'].mean()
#%%
sns.distplot(data['validation_scores_AUC'], kde=False, rug=True)
#%% Differences in AUC are not normal distributed
stats.probplot(data['validation_scores_AUC'], plot=plt)
#%% because for mobilenets difference is bigger!
# sns.scatterplot has no order arg for x axis
data['module'] = data['module'].astype("category").cat.set_categories(order)
data = data.sort_values(['module'])
g = sns.scatterplot(
    x="module", y="validation_scores_AUC", data=data,
    hue='classifier', palette='colorblind', style='cohort_identifier',
)
g.set_xticklabels(order[3:], rotation=80)
plt.gcf().set_size_inches(14, 8.27)
#%%
#%% 512x512 seems a bit better than 2048x2048
# paper a) big w/o protein
data = df_
g = sns.swarmplot(
    x="cohort_identifier", y="validation_scores_AUC",
    hue="modality", dodge=True,
    data=data)
plt.gcf().set_size_inches(14, 8.27)
#%% paired t-test
modalities_paired = df_.query("cohort_identifier != 'proteomics'").set_index(
    ['classifier', 'module', 'modality']
).pivot(columns='cohort_identifier')
statistic, p_value = stats.ttest_rel(
    modalities_paired.loc[
        :, ('validation_scores_AUC', 'ppp1_raw_image_512x512')
    ].values,
    modalities_paired.loc[
        :, ('validation_scores_AUC', 'ppp1_raw_image_2048x2048')
    ].values,
)

data = (modalities_paired.xs(
    'ppp1_raw_image_512x512', level='cohort_identifier', axis=1
) - modalities_paired.xs(
    'ppp1_raw_image_2048x2048', level='cohort_identifier', axis=1
)).reset_index()
p_value, data['validation_scores_AUC'].mean()

#%%
sns.distplot(data['validation_scores_AUC'], kde=False, rug=True)
#%% Differences in AUC are pretty much normal distributed
stats.probplot(data['validation_scores_AUC'], plot=plt)
#%% no pattern over modules
# sns.scatterplot has no order arg for x axis
data['module'] = data['module'].astype("category").cat.set_categories(order)
data = data.sort_values(['module'])
g = sns.scatterplot(
    x="module", y="validation_scores_AUC", data=data,
    hue='classifier', palette='colorblind', style='modality',
)
g.set_xticklabels(order[3:], rotation=80)
plt.gcf().set_size_inches(14, 8.27)

#%% compare discrete metrics relationships
data = df__
hue = 'classifier'
y_vars = ['validation_scores_Brier_Loss', 'validation_scores_Log_Loss',
          'validation_scores_AUC', 'validation_scores_Accuracy']
x_vars = [
    'encoded_features_size',
    'non_varying_features',
    'encoded_image_size_width',
    'raw_image_size_width',
    'module',
]
g = sns.PairGrid(data=data, hue=hue, x_vars=x_vars, y_vars=y_vars,
                 palette='colorblind')
# g = g.map_upper(sns.relplot, col='modality')
# g = g.map_diag(plt.hist)
g = g.map(sns.swarmplot)
g = g.add_legend()

#%% compare metrics relationships
data = df__
y_vars = ['validation_scores_Brier_Loss', 'validation_scores_Log_Loss',
          'validation_scores_AUC', 'validation_scores_Accuracy']
x_vars = [
    'validation_scores_Brier_Loss',
    'validation_scores_DP',
    'validation_scores_F1',
    'validation_scores_Log_Loss',
    'validation_scores_P+',
    'validation_scores_P-',
    'validation_scores_Precision',
    'validation_scores_Recall',
    'validation_scores_Specificity',
    'validation_scores_Youdens',
] + y_vars + [
    'training_scores_mean_test_AUC',
    'training_scores_mean_train_AUC',
    'training_scores_mean_test_Accuracy',
    'training_scores_mean_train_Accuracy',
    'training_scores_mean_test_F1',
    'training_scores_mean_train_F1',
]
g = sns.PairGrid(data=data, hue=hue, x_vars=x_vars, y_vars=y_vars)
# g = g.map_upper(sns.relplot, col='modality')
# g = g.map_diag(plt.hist)
g = g.map(sns.scatterplot)
g = g.add_legend()

#%% AUC (highly correlated to Accuracy and F1) vs Log_Loss
# Brier_Loss and AUC tell same story, just Log_Loss is weird for logistig
# regression; maybe no convergence for some runs in logistic regressions?
data = df__
with sns.axes_style('white'):
    sns.scatterplot(
        y='validation_scores_AUC', x='validation_scores_Log_Loss',
        hue='classifier', style='cohort_identifier', data=data,
        palette='colorblind'
    )
    plt.gcf().set_size_inches(14, 8.27)

#%% AUC vs Brier_Loss
data = df__
with sns.axes_style('white'):
    sns.scatterplot(
        y='validation_scores_Accuracy', x='validation_scores_Brier_Loss',
        hue='classifier', style='cohort_identifier', data=data,
        palette='colorblind'
    )
    plt.gcf().set_size_inches(14, 8.27)

#%%
data = df__
g = sns.catplot(x="cohort_identifier", y="validation_scores_AUC",
                hue="module",
                # dodge=True,
                # row="modality",
                col="classifier",
                data=data, kind="swarm",
                # jitter=True
                )
#%%
# paper 2 main result: encoder importance
data = df__
g = sns.boxplot(x="module", y="validation_scores_AUC", data=data)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=80)
plt.gcf().set_size_inches(14, 8.27)

#%% RandomForrest is consitently worse, XGBoost too but ok with proteomics
# 512x512 better than 2048x2048
# paper 1
data = df__
with sns.axes_style('white'):
    g = sns.scatterplot(
        x="module", y="validation_scores_AUC", data=data,
        hue='classifier', palette='colorblind', style='cohort_identifier',
    )
    g.set_xticklabels(order, rotation=90)
    plt.gcf().set_size_inches(9, 8.27)


#%%
data = df__
g = sns.pointplot(
    x="raw_image_size_width", y="validation_scores_AUC",
    hue="classifier", palette='colorblind',
    # col="module", kind="swarm",
    dodge=True,
    data=data,
)

#%% look at params for best/worst models
param_index_df = pd.io.json.json_normalize(dicts)
param_index_df.columns = [c.replace('.', '_') for c in param_index_df.columns]
# excluded cv_index before
param_index_df = param_index_df[[
    'classifier',
    'cv_index',
    'modality',
    'module',
    'cohort_identifier',
]]


def csv_name(series):
    return '-'.join([
        series["cohort_identifier"],
        series["module"],
        series["modality"],
        series["classifier"],
        'results.csv'
    ])


param_index_df["csv_basename"] = param_index_df.apply(csv_name, axis=1)
#%%
g = sns.catplot(
    x="cv_index",
    col="classifier",
    # hue="classifier",
    palette='colorblind',
    kind="count", sharex=False,
    data=param_index_df,
    facet_kws={'gridspec_kws': {'width_ratios': [2 / 7, 4 / 7, 2 / 7, 1.0]}}
)
axes = g.axes
axes[0, 0].set_xlim(-0.5, 1.5)
axes[0, 1].set_xlim(-0.5, 3.5)
axes[0, 2].set_xlim(-0.5, 1.5)
# axes[0,3].set_xlim(0,11)


#%% [markdown]
"""
### To interpret parameter indexes from param_index_df (last plot)
#### rdf
{'n_estimators': 100, 'random_state': 7899463},
{'n_estimators': 500, 'random_state': 7899463},
#### lgr
{'C': 0.1, 'random_state': 7899463},
{'C': 1.0, 'random_state': 7899463},
{'C': 10.0, 'random_state': 7899463},
{'C': 100.0, 'random_state': 7899463},
#### xgb
{'n_estimators': 100, 'random_state': 7899463},
{'n_estimators': 500, 'random_state': 7899463},
#### svm
{'C': 0.1, 'kernel': 'linear', 'random_state': 7899463},
{'C': 0.1, 'kernel': 'poly', 'random_state': 7899463},
{'C': 0.1, 'kernel': 'rbf', 'random_state': 7899463},
{'C': 1.0, 'kernel': 'linear', 'random_state': 7899463},
{'C': 1.0, 'kernel': 'poly', 'random_state': 7899463},
{'C': 1.0, 'kernel': 'rbf', 'random_state': 7899463},
{'C': 10.0, 'kernel': 'linear', 'random_state': 7899463},
{'C': 10.0, 'kernel': 'poly', 'random_state': 7899463},
{'C': 10.0, 'kernel': 'rbf', 'random_state': 7899463},
{'C': 100.0, 'kernel': 'linear', 'random_state': 7899463},
{'C': 100.0, 'kernel': 'poly', 'random_state': 7899463},
{'C': 100.0, 'kernel': 'rbf', 'random_state': 7899463},
"""

#%%
# for csv, index in param_index_df.set_index("csv_basename")["cv_index"].items():  # noqa
#     print(
#         index,
#         pd.read_csv(os.path.join(DATA_DIR, csv), index_col=0).loc[
#             index, "params"
#         ]
#     )
