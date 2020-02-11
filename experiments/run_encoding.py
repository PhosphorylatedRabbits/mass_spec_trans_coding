"""Encoding of directory of raw ms images.
Writing an xr.DataArray for each modality encoded with each hub module."""
import logging
import os
import re
import sys
import traceback
from collections import OrderedDict

import pandas as pd
import plac
import tensorflow as tf

from mstc.processing import Compose, HubEncoder, Map, PNGReader

assert sys.version_info >= (3, 6)

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


tf.logging.set_verbosity('CRITICAL')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PATTERN = re.compile(
    r'(?P<sample_name>.+?)(\.mzXML\.gz\.image\.0\.)'
    r'(?P<modality>(itms)|(ms2\.precursor=\d{3,}\.\d{2}))'
    r'\.png'
)


def run_all_encodings_on_all_modalities(input_directory, output_directory,
                                        batch_size=4):
    output_directory = os.path.abspath(os.path.expanduser(output_directory))
    assert os.path.exists(output_directory)
    data_dir = os.path.abspath(os.path.expanduser(input_directory))

    sample_set = set()
    modality_set = set()
    for filepath in os.listdir(data_dir):
        groupdict = PATTERN.match(filepath).groupdict()
        sample_set.add(groupdict['sample_name'])
        modality_set.add(groupdict['modality'])

    cohort_identifier = os.path.basename(data_dir)
    glob_patterns = [
        os.path.join(data_dir, f'*{modality}*.png')
        for modality in modality_set
    ]

    modalities_reader = Map(
        PNGReader(directory=data_dir), map_reader='read modalities'
    )

    for module, url in HUB_MODULES.items():
        try:
            logger.info(
                f'{module} encoding starts '
                f'({HUB_MODULES.index.get_loc(module)+1}/{len(HUB_MODULES)})'
            )
            # each encoding of all modalities consumes reader
            # so read again instead of keeping in memory with BroadcastMap
            modalities_encoder = Map(
                HubEncoder(url, batch_size=batch_size,
                           encoder_module_name=module)
            )
            pipeline = Compose(
                [modalities_reader, modalities_encoder],

                pipeline='for encoder, map encoder over all read modalities',
                pipeline_output='single modality, single encoder'
            )

            def is_encoding_required(pattern):
                """function to filter glob_patterns with logging side effect"""
                modality = pattern.split('*')[1]
                if not os.path.exists(os.path.join(
                    output_directory,
                    cohort_identifier + '-' + module + '-' + modality + '.nc'
                )):
                    return True
                else:
                    logger.info(
                        f'skipped modality {modality}, encoding exitst.'
                    )
                    return False
            required_glob_patterns = filter(is_encoding_required, glob_patterns)  # noqa

            for modality_array in pipeline(required_glob_patterns):
                modality = PATTERN.match(
                    modality_array.sample.data[0]
                ).groupdict()['modality']
                name = cohort_identifier + '-' + module + '-' + modality
                modality_array.name = name
                filename = os.path.join(output_directory, name + '.nc')

                modality_array.to_netcdf(filename)
                logger.info(f'{name}.nc was written')
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            logger.warn(f'FAIL with module {module} (url: {url})')
            traceback.print_exc()

    logger.info('Processing done.')


if __name__ == "__main__":
    plac.call(run_all_encodings_on_all_modalities)
