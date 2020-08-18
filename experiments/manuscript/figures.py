#%% [markdown]
"""
Create figures from manuscript.
"""
#%%
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('figures')


def manuscript_sizes(denominator=1, text_width=5.5):
    """
    Figure sizes according to document textwidth.

    Arguments:
        denominator (int): denominator used in the document
        text_width (float): inches of document textwidth
    """

    fig_width = text_width/denominator - (denominator-1)*0.01

    golden_mean = 0.6180339887498949  # Aesthetic ratio (sqrt(5)-1.0)/2.0
    fig_height = fig_width * golden_mean  # height in inches

    return fig_width, fig_height


#%%
if 'get_ipython' in dir():
    # in case interactive session was moved to __file__ dir
    %cd ../..
#%%

DF_PATH = os.path.join('experiments/manuscript/',
                       'classification_results.csv')
swathes_path = 'experiments/manuscript/ms2_precursors.csv'

figure_path = 'data/figures/{}'

df_raw = pd.read_csv(DF_PATH, index_col=0)

#%% [markdown]
"""
# filter and sort DataFrame
"""
#%%
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
df = df_raw[df_raw["module"].apply(lambda x: x not in drop_nets)]

df = df[[
    'classifier',
    'AUC',
    # 'Accuracy',
    # 'F1',
    # 'cv_index',
    # 'training_scores_mean_test_AUC',
    # 'training_scores_mean_test_Accuracy',
    # 'training_scores_mean_test_F1',
    # 'training_scores_mean_train_AUC',
    # 'training_scores_mean_train_Accuracy',
    # 'training_scores_mean_train_F1',
    'encoded_features_size',
    'non_varying_features',
    # 'encoded_image_size',
    'encoded_image_size_height',
    # 'encoded_image_size_width',
    'modality',
    'module',
    # # 'raw_image_size',
    # 'raw_image_size_height',
    'raw_image_size_width',
    'cohort_identifier',
    # # only in rerun
    # 'Brier_Loss',
    # 'DP',
    # 'Log_Loss',
    # 'P+',
    # 'P-',
    # 'Precision',
    # 'Recall',
    # 'Specificity',
    # 'Youdens'
]]

# remove cropped
df = df.query(
    "(raw_image_size_width != '243.0') & (raw_image_size_width != '973.0')"
)

# order is modules sorted for median AUC for all_modalities
module_group = df.query("modality == 'all_modalities'").groupby('module')
module_medians = module_group.median().sort_values(
    by='AUC', ascending=False
)
module_order = module_medians.index

df['module'] = df['module'].astype('category').cat.set_categories(module_order)
df = df.sort_values(['module', 'AUC'], ascending=[True, False])

# only all_modalities
df_ = df.query("modality == 'all_modalities'")
df__ = df.query("modality == 'ms1_only'")
df__['module'].cat.remove_categories(
    ['proteins', 'peptides3', 'peptides4'], inplace=True
)
#%% [markdown]
"""
# plotting settings
"""
#%%
[(key, value) for key, value in plt.rcParamsDefault.items() if 'size' in key]
#%%
[(key, value) for key, value in plt.rcParamsDefault.items() if 'width' in key]

#%%
params = {
    # 'backend': 'ps',
    # 'text.latex.preamble': [r'\usepackage{gensymb}'],
    # 'text.usetex': True,
    # 'font.family': 'serif'
    'axes.labelsize': 5,
    'axes.titlesize': 5,
    'font.size': 5,
    'legend.fontsize': 3,
    'lines.markersize': 2,
    'legend.markerscale': 0.5,
    # 'lines.markeredgecolor': 'None',
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    # 'figure.figsize': [fig_width, fig_height],
    'axes.linewidth': 0.5,
    'boxplot.boxprops.linewidth': 0.5,
    'boxplot.capprops.linewidth': 0.5,
    'boxplot.flierprops.linewidth': 0.5,
    'boxplot.meanprops.linewidth': 0.5,
    'boxplot.medianprops.linewidth': 0.5,
    'boxplot.whiskerprops.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'hatch.linewidth': 0.5,
    'lines.linewidth': 0.5,
    # 'lines.markeredgewidth': 0.1,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'xtick.minor.width': 0.5,
    'ytick.major.width': 0.5,
    'ytick.minor.width': 0.5,
    # 'boxplot.flierprops.markersize': 0.5,
}

matplotlib.rcParams.update(params)

#%% [markdown]
"""
# Figures
"""

#%% paper 2a
#  main result: encoder importance
figure_name = 'median_module.pdf'
fig_width, fig_height = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.boxplot(
    ax=ax, x="module", y="AUC",
    data=df_, flierprops={'markersize': 0.5}
)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=90)

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

#%% like above, but MS1 only (same order as MS2)
figure_name = 'median_module_ms1_only.pdf'
fig_width, fig_height = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.boxplot(
    ax=ax, x="module", y="AUC",
    data=df__, flierprops={'markersize': 0.5}
)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=90)

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')


#%% paper 2b
# RandomForrest is consitently worse, XGBoost too but ok with proteomics
# 512x512 better than 2048x2048
figure_name = 'classifiers_module.pdf'
fig_width, fig_height = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

# with sns.axes_style('white'):
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.scatterplot(
    ax=ax,
    x="module", y="AUC", data=df_,
    hue='classifier', palette='colorblind', style='cohort_identifier',
    markers=['s', 'o', 'v'], edgecolor='face',   # linewidths=0.01  # no change
)
g.set_xticklabels(module_order, rotation=90)

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

#%% like above but MS1 only (same order as MS2)
# RandomForrest is consitently worse, XGBoost too but ok with proteomics
# 512x512 better than 2048x2048
figure_name = 'classifiers_module_ms1_only.pdf'
fig_width, fig_height = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

# with sns.axes_style('white'):
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.scatterplot(
    ax=ax,
    x="module", y="AUC", data=df__,
    hue='classifier', palette='colorblind', style='cohort_identifier',
    markers=['s', 'o', 'v'], edgecolor='face',   # linewidths=0.01  # no change
)

g.set_xticklabels(g.get_xticklabels(), rotation=90)  # [a._text for a in g.get_xticklabels()]

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

#%% paper 3a
# 512x512 seems a bit better than 2048x2048
figure_name = 'resolution_modalities.pdf'
fig_width, fig_height = manuscript_sizes(denominator=3, text_width=5.5)
logger.info(f'{figure_name} sizes: {(2*fig_width, 2*fig_height)}')

# with sns.axes_style('white'):
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.swarmplot(
    ax=ax,
    x="resolution", y="AUC",
    hue="modality", dodge=True,
    data=df.query("cohort_identifier != 'proteomics'").rename(
        columns={'cohort_identifier': 'resolution'}
    ),
    s=2
)
ax.set_xticklabels([
    item.get_text().replace('ppp1_raw_image_', '')
    for item in ax.get_xticklabels()
])
plt.legend(markerscale=0.2)

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')


#%% [markdown]
"""
# paired t-tests (and error distribution)
"""
alpha = 0.001


#%% [markdown]
"""
## ms1 vs all_modalities
"""


#%% pairwise t-test all_modalities vs ms1
modalities_paired = df.query("cohort_identifier != 'proteomics'").set_index(
    ['classifier', 'module', 'cohort_identifier']
).pivot(columns='modality')
statistic, p_value = stats.ttest_rel(
    modalities_paired.loc[:, ('AUC', 'all_modalities')].values,
    modalities_paired.loc[:, ('AUC', 'ms1_only')].values,
)

modalities_delta = (
    modalities_paired.xs('all_modalities', level='modality', axis=1) -
    modalities_paired.xs('ms1_only', level='modality', axis=1)
).reset_index()
print(
    f't: {statistic}\n'
    f'p: {p_value}\n'
    f'H0 (all_modalities smaller or equal) rejected under alpha {alpha}: '
    f'{statistic > 0 and p_value/2 < alpha}\n'
    f'mean_delta (all-ms1) [AUC]: {modalities_delta["AUC"].mean()}'
)


#%% paper 3b
# modalities_delta_dist
figure_name = 'modalities_delta_dist.pdf'
fig_width, fig_height = manuscript_sizes(denominator=3, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

sns.distplot(modalities_delta['AUC'], kde=False, rug=True, ax=ax)
plt.xlabel(u'Δ AUC')
plt.axvline(0, 0, 1)
plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

# Differences in AUC are not normal distributed
# stats.probplot(modalities_delta['AUC'], plot=plt)


#%% because for mobilenets increase more in performance with all_modalities
# compared to ms1 than the other (generally better) modules.

# figure_name = 'modalities_delta_module.pdf'
# fig_width, fig_height = manuscript_sizes(denominator=3, text_width=5.5)
# logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

# fig, ax = plt.subplots(figsize=(fig_width, fig_height))
# # sns.scatterplot has no order arg for x axis, make sure order is correct
# data = modalities_delta.sort_values(['module'])
# g = sns.scatterplot(
#     x="module", y="AUC", data=data,
#     hue='classifier', palette='colorblind', style='cohort_identifier',
#     markers=['o', 'v'], edgecolor='face', linewidths=0.1,
#     ax=ax
# )
# g.set_xticklabels(module_order[3:], rotation=90)
# plt.savefig(figure_path.format(figure_name), bbox_inches='tight')


#%% [markdown]
"""
## difference in image resolution
"""


#%% paired t-test
cohort_paired = df.query("cohort_identifier != 'proteomics'").set_index(
    ['classifier', 'module', 'modality']
).pivot(columns='cohort_identifier')
statistic, p_value = stats.ttest_rel(
    cohort_paired.loc[:, ('AUC', 'ppp1_raw_image_512x512')].values,
    cohort_paired.loc[:, ('AUC', 'ppp1_raw_image_2048x2048')].values,
)

cohort_delta = (cohort_paired.xs(
    'ppp1_raw_image_512x512', level='cohort_identifier', axis=1
) - cohort_paired.xs(
    'ppp1_raw_image_2048x2048', level='cohort_identifier', axis=1
)).reset_index()
print(
    f't: {statistic}\n'
    f'p: {p_value}\n'
    f'H0 (512x512 smaller or equal) rejected under alpha {alpha}: '
    f'{statistic > 0 and p_value/2 < alpha}\n'
    f'mean_delta (512x512-2048x2048) [AUC]: {cohort_delta["AUC"].mean()}'
)


#%% resolution_delta_dist
figure_name = 'resolution_delta_dist.pdf'
fig_width, fig_height = manuscript_sizes(denominator=3, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
sns.distplot(cohort_delta['AUC'], kde=False, rug=True, ax=ax)
plt.xlabel(u'Δ AUC')
plt.axvline(0, 0, 1)
plt.savefig(figure_path.format(figure_name), bbox_inches='tight')
# u'Δ AUC'
# Differences in AUC are pretty much normal distributed
# stats.probplot(cohort_delta['AUC'], plot=plt)


#%% [markdown]
"""
# Encoders table
"""
#%% Table
table = pd.pivot_table(
    df.query("cohort_identifier != 'proteomics'"),
    index=['classifier', 'modality', 'cohort_identifier'],
    columns='module',
    values=['encoded_features_size',
            'encoded_image_size_height',
            'non_varying_features']
)

#%% same for all rows
encoder_characteristics = table.iloc[0].unstack(level=0)[[
    'encoded_image_size_height', 'encoded_features_size'
]]

#%% same for each classifier, pick number from one
varying_features = table.loc['LogisticRegression', 'non_varying_features'].T


#%% percentages
percent_ms1 = varying_features['ms1_only'].divide(
    encoder_characteristics['encoded_features_size'], axis=0
) * 100
percent_ms1.columns = pd.MultiIndex.from_product([['ms1_only'], percent_ms1.columns])

percent_all = varying_features['all_modalities'].divide(
    # with ms2 101 times the total possible features
    encoder_characteristics['encoded_features_size'] * 101, axis=0
) * 100
percent_all.columns = pd.MultiIndex.from_product([['all_modalities'], percent_all.columns])

percent_df = pd.concat([percent_ms1, percent_all], axis=1).astype(int)

#%% merge percent
for column in varying_features.columns:
    varying_features[column] = (
        varying_features[column].map(str)
        + " ("
        + percent_df[column].map(str)
        + '%)'
    )
#%%
encoder_table = pd.concat([
    encoder_characteristics.astype(int), varying_features,
] , axis=1)
encoder_table
#%%
print(encoder_table.to_latex())


#%%
"""
# Swath sizes figure
"""
#%%

ms2 = pd.read_csv(swathes_path) \
      .sort_values('ms2_precursor')


#%%
d = {}
prec_iter = zip(ms2['ms2_precursor'], ms2['ms2_precursor'].diff())
next(prec_iter)  # assume 400 as start
prec = 403.0
down = 3.0
up = 3.0
sw = up + down  # 406.0 - 400.0
to = 406.0
d.update({prec: (np.nan, sw, down, up, to)})
for prec, delta in prec_iter:
    down = prec - to
    up = delta - up
    sw = up + down  # prec + up - to
    to = prec + up
    d.update({prec: (delta, sw, down, up, to)})

swathes = pd.DataFrame(
    d,
    index=['delta', 'swath_size', 'lower_diff', 'upper_diff', 'swath_end']
).T

#%% [markdown]
# the 100 ms2 swathes are not constant, but start with 6, go down to 5
# and then become larger. First: 403 +- 3 Last: 1224.5 +- 24.5

#%% paper 1 swath sizes
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

figure_name = 'swath_sizes.pdf'
fig_height, fig_width = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
sns.scatterplot(
    ax=ax,
    y=swathes.index, x=swathes.swath_size,
    linewidth=0.2
).invert_yaxis()
plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

#%%
