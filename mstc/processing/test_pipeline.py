"""test mstc pipeline on the example of ppp1 ms1 512x512"""
import logging
import os
import shutil
import tempfile
import time
import unittest
from contextlib import contextmanager

import imageio
import numpy as np
import tensorflow as tf

from mstc.processing import (Broadcast, BroadcastMap, Compose, HubEncoder,
                             Map, PNGReader, Stacker)


def silence_imageio_warning(*args, **kwargs):
    """patch image conversion warning"""
    pass


imageio.core.util._precision_warn = silence_imageio_warning

tf.logging.set_verbosity('INFO')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    logger.info(f'[{name}] finished in {int(elapsedTime * 1000)} ms')


class PipelineTestCase(unittest.TestCase):
    """Test reading of images in xr.DataArray."""

    height = 10
    width = 10
    channel = 3
    sample = 33
    hub_host = 'https://tfhub.dev/google/'
    HUB_MODULES = dict([
        ('module0', f'{hub_host}imagenet/nasnet_mobile/feature_vector/1'),
        ('module1', f'{hub_host}imagenet/nasnet_mobile/feature_vector/1'),
    ])
    num_features = 1056
    encoded_image_size = (224, 224)

    def setUp(self):
        """Generate .png images in a temporary folder."""
        self.directory = tempfile.mkdtemp()
        for i in range(self.sample):
            imageio.imwrite(
                os.path.join(self.directory, f'{str(i).zfill(3)}.png'),
                np.random.rand(self.height, self.width, self.channel)
            )
        self.globbing_pattern = os.path.join(
            self.directory, '*.png'
        )
        self.globbing_patterns = [self.globbing_pattern, self.globbing_pattern]

    def test_pipeline_stack_read(self):
        """Test a pipeline where a reader is mapped over multiple patterns
        and the results are stacked"""
        # prep pipline
        reader_map = Map(
            PNGReader(reader='single reader'),
            map_reader='apply each pattern to reader'
        )
        stacker = Stacker(
            dim='pattern',
            stacker='stack images for different patterns'
        )
        pipeline = Compose(
            [reader_map, stacker],
            pipeline='test_pipeline_stack_read'
        )

        # run pipeline
        data_array = pipeline(self.globbing_patterns)

        # checks
        self.assertEqual(
            dict(data_array.sizes),
            dict([
                ('pattern', len(self.globbing_patterns)),
                ('sample', self.sample),
                ('height', self.height),
                ('width', self.width),
                ('channel', self.channel)
            ])
        )
        self.assertEqual(
            data_array.attrs,
            {
                'reader': 'single reader',
                'map_reader': 'apply each pattern to reader',
                'stacker': 'stack images for different patterns',
                'pipeline': 'test_pipeline_stack_read'
            }
        )

    def test_pipeline_stack_encoded(self):
        """Test a pipeline where a read array is broadcast to multiple encoders.
        and the results are stacked"""
        # prep pipline
        reader = PNGReader(reader='single reader')
        with timeit_context('HubEncoders initialization'):
            encoders = (
                HubEncoder(url, encoder_module=module) for
                module, url in self.HUB_MODULES.items()
            )
        encoders_broadcast = Broadcast(
            encoders,
            broadcast_encoders='many encodings of single modality'
        )
        pipeline = Compose(
            [reader, encoders_broadcast],
            pipeline='test_pipeline_unstacked_encoded'
        )
        # run first pipeline (list else we consume generator)
        data_arrays = list(pipeline(self.globbing_pattern))
        data_array_first = data_arrays[0]
        # check
        self.assertEqual(
            dict(data_array_first.sizes),
            dict([
                ('sample', self.sample),
                ('hub_feature', self.num_features),
            ])
        )
        expected_attributes = {
            'reader': 'single reader',
            'encoder_module': 'module0',
            'encoded_image_size': self.encoded_image_size,
            'broadcast_encoders': 'many encodings of single modality',
            'pipeline': 'test_pipeline_unstacked_encoded'
        }
        self.assertEqual(
            data_array_first.attrs,
            expected_attributes
        )

        # continue with other pipeline
        stacker = Stacker(
            dim='encoder', stacker='stack images for different encoders'
        )
        data_array = Compose(
            [stacker],
            pipeline='test_pipeline_stack_encoded'
        )(data_arrays)

        # check
        self.assertEqual(
            dict(data_array.sizes),
            dict([
                ('encoder', len(self.HUB_MODULES)),
                ('sample', self.sample),
                ('hub_feature', self.num_features),
            ])
        )
        expected_attributes.update({
            'stacker': 'stack images for different encoders',  # new
            'pipeline': 'test_pipeline_stack_encoded'  # updated
        })
        # 'encoded_image_size' is the same for both modules
        del expected_attributes['encoder_module']  # stacker value not equal
        self.assertEqual(
            data_array.attrs,
            expected_attributes
        )

    def test_pipeline_map_broadcast(self):
        reader_map = Map(
            PNGReader(reader='read imgages from pattern'),
            map_reader='apply each pattern ("modality") to reader'
        )
        with timeit_context('HubEncoders initialization'):
            encoders = [  # reusing with map, cannot be generator
                HubEncoder(url, encoder_module=module) for
                module, url in self.HUB_MODULES.items()
            ]
        modality_encoders = Broadcast(
            encoders,
            broadcast_encoders='many encodings of single modality'
        )
        stacker = Stacker(
            dim='hub_modules',
            encoded_stacker='aggregate different hub module encodings'
        )
        modalities_all_encoder = Map(
            Compose((modality_encoders, stacker)),
            map_all_encoder='apply (all) encoders to each modality'
        )
        pipeline = Compose(
            (reader_map, modalities_all_encoder, Stacker(dim='modality')),
            pipeline='read all modalities, then encode each with all encoders'
        )

        data_array = pipeline(self.globbing_patterns)

        print(data_array)
        # TODO check result

    def test_pipeline_BroadcastMap(self):
        """run each encoder on all modalities (first)"""
        reader_map = Map(
            PNGReader(reader='read imgages from pattern'),
            map_reader='apply each pattern ("modality") to reader'
        )
        with timeit_context('HubEncoders initialization'):
            encoders = (
                HubEncoder(url, encoder_module=module) for
                module, url in self.HUB_MODULES.items()
            )
        modalities_encoders = BroadcastMap(
            encoders,
            broadcast_encoders='apply (all) encoders to each modality'
        )
        stacker = Stacker(
            dim='encoded_x_modality',
            encoded_stacker='aggregate all encodings for all modalities'
        )
        pipeline = Compose(
            (reader_map, modalities_encoders, stacker),
            pipeline='read all modalities, then encode all with each encoders'
        )

        data_array = pipeline(self.globbing_patterns)

        print(data_array)
        # TODO check result

    def tearDown(self):
        """Delete .png images and the temporary folder."""
        shutil.rmtree(self.directory)
