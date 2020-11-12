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

from seaborn.miscplot import palplot

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

METRIC = 'AUC'
df = df[[
    'classifier',
    METRIC,
    # 'AUC',
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


# architecture groups
def assign_module(module_name):
    module_group = module_name.split('_')[0]
    if 'nasnet' in module_group or module_group == 'amoebanet':
        module_group = 'archirecture search'
    if module_group in ['proteins', 'peptides3', 'peptides4']:
        module_group = 'proteomic'
    return module_group


df['architecture'] = df['module'].apply(assign_module)


#%% order is modules sorted for median AUC for all_modalities
module_group = df.query("modality == 'all_modalities'").groupby('module')[METRIC]
module_medians = module_group.median().sort_values(ascending=False)
module_order = module_medians.index


df['module'] = df['module'].astype('category').cat.set_categories(module_order)
df = df.sort_values(['module', METRIC], ascending=[True, False])

# only all_modalities
df_ = df.query("modality == 'all_modalities'")
df__ = df.query("modality == 'ms1_only'")
# no ['proteins', 'peptides3', 'peptides4'] for ms1_only
df__['module'].cat.remove_unused_categories(inplace=True)


#%% [markdown]
"""
# plotting settings
"""
#%% colors
archs = df['architecture'].unique()
architecture_colors = pd.Series(
    sns.color_palette(n_colors=len(archs)), index=archs
)
module_colors = pd.Series(
    module_order, index=module_order
).map(lambda x: architecture_colors[assign_module(x)])

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
Numbers & Tables
"""
# %%
all_auc_pivot = np.round(
    df, decimals=3
).replace(to_replace=r'^ppp1_raw_image_', value='', regex=True)[
    ['classifier', 'module', 'cohort_identifier', 'AUC', 'modality']
].set_index(
    ['module', 'cohort_identifier']
).pivot(columns=['modality', 'classifier'])

print(all_auc_pivot.to_latex())

# %% same as above but make maximum value per module bold
all_auc_pivot = np.round(
    df, decimals=3
).replace(to_replace=r'^ppp1_raw_image_', value='', regex=True)[
    ['classifier', 'module', 'cohort_identifier', 'AUC', 'modality']
].set_index(
    ['module']
).pivot(
    columns=['modality', 'classifier', 'cohort_identifier']
).mask(
    cond=lambda df: df.eq(df.max(axis=1), axis=0),
    other=lambda df: df.applymap(lambda x: f'\textbf{{{x:.3f}}}'),
    axis=1
).stack()
# all_auc_pivot
print(all_auc_pivot.to_latex(escape=False))
# %%
modalities_paired = df.set_index(
    ['classifier', 'module', 'cohort_identifier', 'architecture']
).pivot(columns='modality')

paired_module_groups = modalities_paired.groupby('module')
module_stats_df = pd.concat([
    paired_module_groups.median().loc[:, [METRIC]].rename(columns={METRIC: f'median {METRIC}'}),
    paired_module_groups.mean().loc[:, [METRIC]].rename(columns={METRIC: f'mean {METRIC}'}),
    paired_module_groups.std().loc[:, [METRIC]].rename(columns={METRIC: f'standard deviation {METRIC}'}),
], axis=1)
module_stats_df['architecture'] = module_stats_df.index.map(assign_module)

#%% max performance
print('best overall\n', df.loc[df[METRIC].idxmax()])
no_prot = df.query("module not in ['proteins', 'peptides3', 'peptides4']")
no_prot['module'].cat.remove_unused_categories(inplace=True)
print('best off-the-shelf\n', no_prot.loc[no_prot[METRIC].idxmax()])
print('best off-the-shelf ms1_only\n', df__.loc[df__[METRIC].idxmax()])
print(module_stats_df.to_latex())
module_stats_df
#%% MS1 vs all rank correlation
# all values
correlation_df = modalities_paired.query(
    "cohort_identifier != 'proteomics'"
)[METRIC]
print(
    "all values Spearman's rank correlation: ",
    f'{correlation_df.corr(method="spearman").iloc[0, 1]}'
)
# median
print(
    "module median Spearman's rank correlation: ",
    f'{correlation_df.groupby("module").median().corr(method="spearman").iloc[0, 1]}'
)

# %% random forrest vs XGBoost
classifiers_paired = df.set_index(
    ['modality', 'module', 'cohort_identifier', 'architecture']
).pivot(columns='classifier')[METRIC]

trees_diff = classifiers_paired['XGBoost'] - classifiers_paired['RandomForest']
print(trees_diff.mean())
print(trees_diff.std())

ax = sns.distplot(trees_diff,)
for arch, color in architecture_colors.iteritems():
    # multicolored rug
    sns.rugplot(
        trees_diff[trees_diff.index.get_level_values(3) == arch],
        ax=ax, color=color)
# cls_rank_correlations = classifiers_paired.corr(method="spearman")


#%% [markdown]
"""
# Figures
"""
#%% ms1 vs all auc correlation all_values
figure_name = 'correlation.pdf'
fig_width, fig_height = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.scatterplot(
    ax=ax, x='all_modalities', y='ms1_only',
    data=correlation_df.reset_index(),
    hue='architecture', palette=architecture_colors.values[1:],
    style='cohort_identifier',
    markers=['o', 'v'],
    edgecolor='face',
)
g.set(xlabel=f'all_modalities [{METRIC}]', ylabel=f'ms1_only [{METRIC}]')
plt.axis('equal')
plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

#%%
figure_name = 'correlation_medians.pdf'
fig_width, fig_height = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.scatterplot(
    ax=ax, x='all_modalities', y='ms1_only',
    data=module_stats_df[f'median {METRIC}'][3:],
    hue=module_stats_df['architecture'][3:],
    palette=architecture_colors.values[1:],
    edgecolor='face',
)
g.set(xlabel=f'all_modalities [{METRIC}]', ylabel=f'ms1_only [{METRIC}]')
plt.axis('equal')
plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

#%% combining figure 2
figure_name = 'module_overlayed.pdf'
fig_width, fig_height = manuscript_sizes(denominator=1, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, g = plt.subplots(figsize=(fig_width, fig_height))
# box
g = sns.boxplot(
    ax=g, x="module", y=METRIC,
    hue='architecture', dodge=False, palette=architecture_colors.values,
    data=df_,
    fliersize=0.0  # outliers shown by scatterplot
)
# points
clfs_cols = sns.color_palette("Dark2", 8, 1.0)[3:-1]
g = sns.scatterplot(
    ax=g,
    x="module", y=METRIC, data=df_,
    hue='classifier', palette=clfs_cols, style='cohort_identifier',
    markers=['s', 'o', 'v'], linewidth=0.1,  edgecolor="grey",
    # y_jitter=True  # non-functional in sns
)
# legend add 'architecture' subtitle
handles, lables = g.get_legend_handles_labels()
empty = matplotlib.patches.Rectangle(
    (0, 0), 0, 0,
    fill=False, edgecolor='none', visible=False
)
handles.insert(0, empty)
lables.insert(0, 'architecture')
g.legend(handles, lables)

# xticks
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=90)

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')
#%% combining figure 2 but MS1
figure_name = 'module_overlayed_ms1.pdf'
fig_width, fig_height = manuscript_sizes(denominator=1, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, g = plt.subplots(figsize=(fig_width, fig_height))
# box
g = sns.boxplot(
    ax=g, x="module", y=METRIC,
    hue='architecture', dodge=False, palette=architecture_colors.values[1:],
    data=df__,
    fliersize=0.0  # outliers shown by scatterplot
)
# points
clfs_cols = sns.color_palette("Dark2", 8, 1.0)[3:-1]
g = sns.scatterplot(
    ax=g,
    x="module", y=METRIC, data=df__,
    hue='classifier', palette=clfs_cols, style='cohort_identifier',
    markers=['o', 'v'], linewidth=0.1,  edgecolor="grey",
    # y_jitter=True  # non-functional in sns
)
# legend add 'architecture' subtitle
handles, lables = g.get_legend_handles_labels()
empty = matplotlib.patches.Rectangle(
    (0, 0), 0, 0,
    fill=False, edgecolor='none', visible=False
)
handles.insert(0, empty)
lables.insert(0, 'architecture')
g.legend(handles, lables)

# xticks
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=90)

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')
#%% paper 2a
#  main result: encoder importance
figure_name = 'median_module.pdf'
fig_width, fig_height = manuscript_sizes(denominator=2, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
g = sns.boxplot(
    ax=ax, x="module", y=METRIC,
    hue='architecture', dodge=False, palette=architecture_colors.values,
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
    ax=ax, x="module", y=METRIC,
    hue='architecture', dodge=False, palette=architecture_colors.values[1:],
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
    x="module", y=METRIC, data=df_,
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
    x="module", y=METRIC, data=df__,
    hue='classifier', palette='colorblind', style='cohort_identifier',
    markers=['s', 'o', 'v'], edgecolor='face',   # linewidths=0.01  # no change
)

g.set_xticklabels(g.get_xticklabels(), rotation=90)  # [a._text for a in g.get_xticklabels()]

plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

#%% paper 3a
# 512x512 seems a bit better than 2048x2048
figure_name = 'resolution_modalities.pdf'
fig_width, fig_height = manuscript_sizes(denominator=3, text_width=5.5)
logger.info(f'{figure_name} sizes: {(2*fig_width, fig_height)}')

# with sns.axes_style('white'):
# fig, ax = plt.subplots(figsize=(2*fig_width, fig_height))

g = sns.catplot(
    kind="swarm",
    x="resolution", y=METRIC,
    hue='architecture', palette=architecture_colors.values[1:],
    col="modality",
    data=df.query("cohort_identifier != 'proteomics'").rename(
        columns={'cohort_identifier': 'resolution'}
    ),
    s=3,
    height=2*fig_height, aspect=(1.618/3),  # width=fig_width
    # legend=False,
    legend_out=True,
    facet_kws={'gridspec_kws': {'wspace': 0.0001}}
)
for axes in g.axes.flat:
    # save space in axes title
    axes.set_title(axes.get_title().split(' ')[-1])
    # save space in x labels
    axes.set_xticklabels([
        item.get_text().replace('ppp1_raw_image_', '')
        for item in axes.get_xticklabels()
    ])

# plt.gcf().set_size_inches(2*fig_width, fig_height)
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
modalities_paired_no_prot = modalities_paired.query(
    "cohort_identifier != 'proteomics'"
)
statistic, p_value = stats.ttest_rel(
    modalities_paired_no_prot.loc[:, (METRIC, 'all_modalities')].values,
    modalities_paired_no_prot.loc[:, (METRIC, 'ms1_only')].values,
)

modalities_delta = (
    modalities_paired_no_prot.xs('all_modalities', level='modality', axis=1) -
    modalities_paired_no_prot.xs('ms1_only', level='modality', axis=1)
).reset_index()
print(
    f't: {statistic}\n'
    f'p: {p_value}\n'
    f'H0 (all_modalities smaller or equal) rejected under alpha {alpha}: '
    f'{statistic > 0 and p_value/2 < alpha}\n'
    f'mean_delta (all-ms1) [{METRIC}]: {modalities_delta[METRIC].mean()}'
)


#%% paper 3b
# modalities_delta_dist
figure_name = 'modalities_delta_dist.pdf'
fig_width, fig_height = manuscript_sizes(denominator=3, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

sns.distplot(modalities_delta[METRIC], kde=False, ax=ax)  #  rug=True
for arch, color in architecture_colors.iteritems():
    # multicolored rug
    sns.rugplot(
        modalities_delta.query('architecture == @arch')[METRIC],
        ax=ax, color=color)
plt.xlabel(f'Δ {METRIC}')
plt.axvline(0, 0, 1)
plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

# Differences in AUC are not normal distributed
# stats.probplot(modalities_delta[METRIC], plot=plt)


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
    ['classifier', 'module', 'modality', 'architecture']
).pivot(columns='cohort_identifier')
statistic, p_value = stats.ttest_rel(
    cohort_paired.loc[:, (METRIC, 'ppp1_raw_image_512x512')].values,
    cohort_paired.loc[:, (METRIC, 'ppp1_raw_image_2048x2048')].values,
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
    f'mean_delta (512x512-2048x2048) [{METRIC}]: {cohort_delta[METRIC].mean()}'
)


#%% resolution_delta_dist
figure_name = 'resolution_delta_dist.pdf'
fig_width, fig_height = manuscript_sizes(denominator=3, text_width=5.5)
logger.info(f'{figure_name} sizes: {(fig_width, fig_height)}')

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
sns.distplot(cohort_delta[METRIC], kde=False, ax=ax)  #  rug=True
for arch, color in architecture_colors.iteritems():
    # multicolored rug
    sns.rugplot(
        cohort_delta.query('architecture == @arch')[METRIC],
        ax=ax, color=color)
plt.xlabel(f'Δ {METRIC}')
plt.axvline(0, 0, 1)
plt.savefig(figure_path.format(figure_name), bbox_inches='tight')

# Differences in AUC are pretty much normal distributed
# stats.probplot(cohort_delta[METRIC], plot=plt)


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
], axis=1)
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
