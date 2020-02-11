"""Use encoded features for specific classification task."""
import glob
import json
import logging
import os
import re
import sys
import warnings
from collections import OrderedDict
from functools import partial

import pandas as pd
import numpy as np
import plac
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    brier_score_loss, log_loss,
    confusion_matrix, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

from mstc.learning import generate_cross_validation_pipeline
from mstc.processing import Flatten, Stacker

assert sys.version_info >= (3, 6)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HUB_MODULES = pd.Series(OrderedDict([
    # 1-10
    ('inception_v3_imagenet', 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),  # noqa
    # # ('mobilenet_v2', 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2')  # noqa
    ('mobilenet_v2_100_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'),  # noqa
    ('inception_resnet_v2', 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1'),  # noqa
    ('resnet_v2_50', 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1'),  # noqa
    ('resnet_v2_152', 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1'),  # noqa
    ('mobilenet_v2_140_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2'),  # noqa
    ('pnasnet_large', 'https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2'),  # noqa
    ('mobilenet_v2_035_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2'),  # noqa
    ('mobilenet_v1_100_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1'),  # noqa
    # 11-20
    ('mobilenet_v1_050_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/1'),  # noqa
    ('mobilenet_v2_075_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/2'),  # noqa
    # # ('inception_v3', 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2')  # noqa
    ('resnet_v2_101', 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1'),  # noqa
    # # ('quantops', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1'),  # noqa
    ('nasnet_large', 'https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1'),  # noqa
    ('mobilenet_v2_100_96', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/2'),  # noqa
    ('inception_v1', 'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1'),  # noqa
    ('mobilenet_v2_035_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/2'),  # noqa
    ('mobilenet_v2_050_224', 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/2'),  # noqa
    # 21-30
    ('mobilenet_v2_100_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2'),  # noqa
    ('nasnet_mobile', 'https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1'),  # noqa
    ('inception_v3_inaturalist', 'https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1'),  # noqa
    ('mobilenet_v1_025_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/1'),  # noqa
    ('mobilenet_v2_050_128', 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/2'),  # noqa
    ('inception_v2', 'https://tfhub.dev/google/imagenet/inception_v2/feature_vector/1'),  # noqa
    ('mobilenet_v1_025_224', 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/1'),  # noqa
    ('mobilenet_v2_075_96', 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/2'),  # noqa
    ('mobilenet_v1_100_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/1'),  # noqa
    ('mobilenet_v1_050_128', 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/1'),  # noqa
    # other
    ('amoebanet_a_n18_f448', 'https://tfhub.dev/google/imagenet/amoebanet_a_n18_f448/feature_vector/1'),  # noqa
]))

PATTERN = re.compile(
    r'(?P<sample_name>.+?)(\.mzXML\.gz\.image\.0\.)'
    r'(?P<modality>(itms)|(ms2\.precursor=\d{3,}\.\d{2}))'
    r'\.png'
)

# collected from NotUniqueError on calling `match_samplename`
DUPLICATE_PPPB_ID = {
    'guot_PC1_170127_CPP24_sw': ['PPPB534', 'PPPB1071'],
    'guot_pc4_161121_sw_2ug_1': ['PPPB449', 'PPPB465'],
    'guot_PC3_170202_CPP274_sw': ['PPPB663', 'PPPB1252'],
    'guot_pc3_161121_sw_2ug_1': ['PPPB463', 'PPPB919'],
    'guot_PC1_170127_CPP22_sw': ['PPPB625', 'PPPB647'],
    'guot_PC2_170213_CPP109_sw': ['PPPB82', 'PPPB1159'],
    'guot_PC1_170127_CPP23_sw': ['PPPB1306', 'PPPB1534'],
    'guot_pc4_161121_sw_2ug_3': ['PPPB243', 'PPPB681'],
    'guot_PC2_170213_PCPOOL4_sw': ['PPPB623', 'PPPB1230'],
    'guot_PC2_170213_CPP107_sw': ['PPPB939', 'PPPB1067'],
    'guot_PC3_170202_CPP279_sw': ['PPPB1016', 'PPPB1417'],
    'guot_PC3_170202_CPP282_sw': ['PPPB234', 'PPPB1348'],
    'guot_PC3_170202_CPP281_sw': ['PPPB436', 'PPPB473'],
    'guot_PC3_170202_CPP275_sw': ['PPPB541', 'PPPB1115'],
    'guot_PC1_170127_CPP130_sw': ['PPPB224', 'PPPB1519'],
    'guot_PC3_170202_CPP276_sw': ['PPPB1083', 'PPPB1242'],
    'guot_PC2_170213_CPP108_sw': ['PPPB320', 'PPPB992'],
    'guot_pc3_161121_sw_2ug_3': ['PPPB226', 'PPPB438'],
    'guot_PC3_170202_CPP284_sw': ['PPPB802', 'PPPB898'],
    'guot_PC3_170202_CPP278_sw': ['PPPB207', 'PPPB1269'],
    'guot_PC1_170127_CPP129_sw': ['PPPB805', 'PPPB1030'],
    'guot_PC1_170127_CPP128_sw': ['PPPB777', 'PPPB1384'],
    'guot_PC3_170202_CPP280_sw': ['PPPB1442', 'PPPB1467'],
    'guot_PC3_170202_CPP283_sw': ['PPPB1091', 'PPPB1113']
}


class NotUniqueError(Exception):
    pass


def subdict(dictionary, keys):
    return {k: dictionary[k] for k in keys if k in dictionary}


def sizedict(sizes):
    return {'height': int(sizes[0]), 'width': int(sizes[1])}


def match_samplename(samplename, pppb_mapping):
    """translate samplename to pppb id
    treat badly behaved samplenames and duplicates"""
    # correct common mistakes
    samplename = samplename.upper().replace(
        '__', '_'
    ).replace(
        '_PC3_170220_CP', '_PC3_170220_CPP'
    )
    # badly placed "PC6" (no such files as of now)?
    mapped = pppb_mapping[samplename]

    # there are duplicates pppb for some files, these should be removed earlier
    # see DUPLICATE_PPPB_ID

    if isinstance(mapped, str):
        return mapped
    else:
        raise NotUniqueError(
            f'{samplename} has multiple PPPB_IDs: {mapped.values}'
        )
        # protein expression vectors correlate extremely between them
        # but some of these are actually all NaN, such are always first of pair


def read_expression(expression_dir):
    """Read expression files, clean and save in coherent manner.
    """
    def join_dir(basename):
        return os.path.join(expression_dir, basename)

    proteins_df = pd.read_csv(join_dir('2018-05-21pppb_prot.txt'),
                              "\t").set_index(["prot", "tg"]).T
    proteins_df.index = proteins_df.index.map(str.upper)

    peptides2_df = pd.read_csv(join_dir('pppb_pep2.zip'), "\t", index_col=0).T
    peptides3_df = pd.read_csv(join_dir('pppb_pep3.zip'), "\t", index_col=0).T
    peptides4_df = pd.read_csv(join_dir('2018-05-21 pppb_pep4.zip'),
                               "\t", index_col=0).T
    peptides2_df.index = peptides2_df.index.map(str.upper)
    peptides3_df.index = peptides3_df.index.map(str.upper)
    peptides4_df.index = peptides4_df.index.map(str.upper)

    return {
        'proteins': proteins_df,
        'peptides2': peptides2_df,
        'peptides3': peptides3_df,
        'peptides4': peptides4_df,
    }


# results in final table in multiindex with levels:
# - ms1 vs all modalities
# - encoding
# - size
# - classifier

RANDOM_STATE = 7899463
SCORING = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy',
        'F1': 'f1'
    }


def compute_scores(y, X, classifier):
    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    assert recall == recall_score(y, y_pred)

    return {
        # proba
        'AUC': roc_auc_score(y, y_proba),
        'Brier_Loss': brier_score_loss(y, y_proba),
        'Log_Loss': log_loss(y, y_proba),
        # pred
        'Accuracy': accuracy_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'Youdens': sensitivity - (1 - specificity),
        'P+': sensitivity / (1 - specificity),
        'P-': (1 - sensitivity) / specificity,
        'DP': np.sqrt(3) / np.pi * (
            np.log(sensitivity / (1 - sensitivity))
            + np.log(specificity / (1 - specificity))
        ),
        'Recall': recall,
        'Precision': precision,
        'Specificity': specificity,
    }


PARAMETER_GRID = {
    'C': [1e-1, 1e0, 1e1, 1e2],
    'kernel': ['linear', 'poly', 'rbf'],
    'n_estimators': [100, 500],
}


@plac.annotations(
    all_modalities=plac.Annotation(
        'use not only ms1 but also all ms2',
        'flag', 'all'),
    cohort_identifier=plac.Annotation(
        'cohort_directory of original image data',
        'option', choices=[
            'ppp1_raw_image_2048x2048', 'ppp1_raw_image_512x512',
            'ppp1_cropped_image_2048x973', 'ppp1_cropped_image_512x243'
        ]),
    module=plac.Annotation(
        'name of single TF Hub module used for encoding '
        'or "expressions" (default: all)',
        'option'),
    classifier=plac.Annotation(
        'name of single classifier to use for classification (default: all)',
        'option'),
    # ['all', 'LogisticRegression', 'SVC', 'RandomForest', 'XGBoost']
    n_jobs=plac.Annotation(
        'number of parallel jobs in GridSearch (default: 1)',
        'option', type=int),
)
def run_all_encodings_on_all_modalities(
    annotation_csv, index_csv, expression_directory,
    encoded_directory, output_directory,
    all_modalities, cohort_identifier,
    module='all', classifier='all',
    n_jobs=1
):
    module_selection = module
    classifier_selection = classifier
    output_directory = os.path.abspath(os.path.expanduser(output_directory))
    assert os.path.exists(output_directory)
    data_dir = os.path.abspath(os.path.expanduser(encoded_directory))

    # annotation
    annotation_df = pd.read_csv(
        os.path.abspath(os.path.expanduser(annotation_csv)),
        index_col=0
        ).set_index("PPPB_ID")
    pppb_mapping = annotation_df.reset_index().set_index("Raw_ID")["PPPB_ID"]
    # ordered selection of valid samples
    y = pd.read_csv(
        os.path.abspath(os.path.expanduser(index_csv)),
        index_col=0, squeeze=True
    )
    index = y.index
    assert all(
        y == (annotation_df.loc[index]['Tissue'] == "T")
    )
    # This index was prepared in the following way:
    # only use set of pppb available in all datatypes, i.e. intersection of images and protein and peptide expressions (from read_expression())  # noqa
    # minus set of pppb to remove (other labels, duplicate ids)

    cohorts_set, module_set, modality_set = set(), set(), set()
    for filepath in os.listdir(data_dir):
        cohort_directory, module, modality = filepath[:-3].split('-')   # '.nc'
        cohorts_set.add(cohort_directory)
        module_set.add(module)
        modality_set.add(modality)

    if cohort_identifier not in cohorts_set:
        raise OSError(
            f'No encodings for {cohort_identifier} found under {data_dir}'
        )

    # classifiers
    classifier_pipeline = partial(
        generate_cross_validation_pipeline,
        folds=6,
        repeats=2,
        random_state=RANDOM_STATE,
        number_of_jobs=n_jobs,
        scoring=SCORING,
        refit='AUC',
    )

    classifiers = {
        'LogisticRegression': classifier_pipeline(
            LogisticRegression(solver='lbfgs', max_iter=300),
            subdict(PARAMETER_GRID, ['C']),
        ),
        'SVC': classifier_pipeline(
            SVC(gamma='auto', probability=True),
            subdict(PARAMETER_GRID, ['C', 'kernel']),
        ),
        'RandomForest': classifier_pipeline(
            RandomForestClassifier(),
            subdict(PARAMETER_GRID, ['n_estimators']),
        ),
        'XGBoost': classifier_pipeline(
            XGBClassifier(),
            subdict(PARAMETER_GRID, ['n_estimators']),
        ),
    }
    if classifier_selection != 'all':
        classifiers = {classifier_selection: classifiers[classifier_selection]}

    make_expression_iterator = False
    if module_selection == 'all':
        module_iterator = HUB_MODULES.index
        make_expression_iterator = True
    elif module_selection == 'expressions':  # skip to expressions
        module_iterator = []
        make_expression_iterator = True
    else:
        if module_selection not in module_set:
            raise OSError(
                f'No encodings for {module_selection} found under {data_dir}'
            )
        if module_selection not in HUB_MODULES.keys():
            raise KeyError(
                f'Module {module_selection} not known for encodings.'
            )
        module_iterator = [module_selection]

    for module in module_iterator:
        logger.info(f'classification for {module} starts')
        modality = '*' if all_modalities else 'itms'

        glob_pattern = f'{cohort_identifier}-{module}-{modality}.nc'
        encoded_modalities = []
        for filepath in glob.glob(os.path.join(data_dir, glob_pattern)):
            encoded_array = xr.open_dataarray(filepath)
            modality = encoded_array.name.split('-')[2]
            # reduce coords from filename to sample name
            sample_index = [
                PATTERN.match(sample_filename).groupdict()['sample_name']
                .split('/')[-1]
                for sample_filename in encoded_array.indexes['sample']
            ]
            encoded_array = encoded_array.assign_coords(sample=sample_index)

            encoded_modalities.append((modality, encoded_array))

        n_modalities = len(encoded_modalities)
        if all_modalities and n_modalities != 101:
            logger.warning(
                f'{module} has only {n_modalities}/101 modalities available'
            )
            module += '_incomplete'
        encoded_features_size = encoded_modalities[0][1].sizes['hub_feature']

        encoded_modalities.sort(key=lambda key_value: key_value[0])
        encoded_coords, encoded_modalities = zip(*encoded_modalities)
        encoded_module = Stacker(dim='modality', axis=1)(
            list(encoded_modalities)).assign_coords(
            modality=list(encoded_coords))
        del encoded_modalities

        # drop samples with multiple pppb before renaming
        encoded_module = encoded_module.drop(
            list(DUPLICATE_PPPB_ID.keys()), dim='sample'
        )
        pppb_index = [
            match_samplename(sample, pppb_mapping)
            for sample in encoded_module.indexes['sample']
        ]
        encoded_module = encoded_module.assign_coords(sample=pppb_index)
        encoded_module = Flatten(dim_to_keep='sample')(encoded_module)

        X_train, X_test, y_train, y_test = train_test_split(
            encoded_module.loc[index].values, y,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=y
        )
        encoded_image_size = sizedict(
            encoded_module.attrs['encoded_image_size']
        )
        del encoded_module

        for classifier, pipeline in classifiers.items():
            modality = 'all_modalities' if all_modalities else 'ms1_only'
            name = '-'.join(
                [cohort_identifier, module, modality, classifier, 'results'])
            cv_path = os.path.join(output_directory, name + '.csv')
            json_path = os.path.join(output_directory, name + '.json')
            # check if trained results available already
            if os.path.exists(json_path):
                logger.info(f'skipping existing {name}')
                continue
            else:
                logger.info(f'computing {name}')

            # train
            pipeline.fit(X_train, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_df = pd.DataFrame(pipeline.steps[2][1].cv_results_)
            cv_index = int(pipeline.steps[2][1].best_index_)
            training_scores = cv_df.loc[cv_index, [
                'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1',
                'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_F1'
            ]].to_dict()
            # run valitation
            validation_scores = compute_scores(y_test, X_test, pipeline)
            # collect results
            results = {
                'raw_image_size':
                    sizedict(cohort_identifier.split('_')[-1].split('x')),
                'encoded_image_size':
                    encoded_image_size,
                'encoded_features_size': encoded_features_size,
                'non_varying_features':
                    int(sum(pipeline.steps[0][1]._get_support_mask())),
                'cohort_identifier': cohort_identifier,
                'module': module,
                'modality': modality,
                'classifier': classifier,
                'cv_index': cv_index,
                'training_scores': training_scores,
                'validation_scores': validation_scores,
            }
            # write to disk
            cv_df.to_csv(cv_path)
            with open(json_path, 'w') as open_file:
                json.dump(results, open_file)
            logger.info(f'{name}: {validation_scores}')

    if make_expression_iterator:
        expression_dict = read_expression(
            os.path.abspath(os.path.expanduser(expression_directory))
        )
        expression_iterator = expression_dict.items()
    else:   # skip expressions
        expression_iterator = {}.items()
    modality = 'all_modalities'
    cohort_identifier = 'proteomics'
    for module, expression_df in expression_iterator:
        expression_df = expression_df.loc[index].dropna(axis=1)
        if expression_df.shape[1] == 0:
            logger.critical(
                f'No classification of {module}; only features with Na'
            )
            continue
        else:
            logger.info(f'classification for {module} starts')
        # y = annotation_df.loc[index, "Tissue"] == "T"
        X_train, X_test, y_train, y_test = train_test_split(
            expression_df.values, y,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=y
        )
        for classifier, pipeline in classifiers.items():
            name = '-'.join(
                [cohort_identifier, module, modality, classifier, 'results'])
            cv_path = os.path.join(output_directory, name + '.csv')
            json_path = os.path.join(output_directory, name + '.json')
            # check if trained results available already
            if os.path.exists(json_path):
                logger.info(f'skipping existing {name}')
                continue
            else:
                logger.info(f'computing {name}')
            # train
            try:
                pipeline.fit(X_train, y_train)
            except ValueError:
                logger.critical(
                    f'no classification of {module} with shape {expression_df.shape}'  # noqa
                )
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_df = pd.DataFrame(pipeline.steps[2][1].cv_results_)
            cv_index = int(pipeline.steps[2][1].best_index_)
            training_scores = cv_df.loc[cv_index, [
                'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_F1',
                'mean_train_AUC', 'mean_train_Accuracy', 'mean_train_F1'
            ]].to_dict()
            # run valitation
            validation_scores = compute_scores(y_test, X_test, pipeline)
            # collect results
            results = {
                'raw_image_size': None,
                'encoded_image_size': None,
                'encoded_features_size': expression_df.shape[1],
                'non_varying_features':
                    int(sum(pipeline.steps[0][1]._get_support_mask())),
                'cohort_identifier': cohort_identifier,
                'module': module,
                'modality': modality,
                'classifier': classifier,
                'cv_index': cv_index,
                'training_scores': training_scores,
                'validation_scores': validation_scores,
            }
            # write to disk
            cv_df.to_csv(cv_path)
            with open(json_path, 'w') as open_file:
                json.dump(results, open_file)
            logger.info(f'{name}: {validation_scores}')
    logger.info('Processing done.')


if __name__ == "__main__":
    plac.call(run_all_encodings_on_all_modalities)
