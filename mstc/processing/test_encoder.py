"""Tests for data encoders."""
import logging
import unittest

import numpy as np
import tensorflow as tf
import xarray as xr

from .encoder import Flatten, HubEncoder

tf.logging.set_verbosity('INFO')
logging.getLogger(
    'processing.encoder'
).setLevel(logging.DEBUG)


def coord_mapping(name):
    """Returns alternative coordinate."""
    return name + 100


class HubEncoderRGBTestCase(unittest.TestCase):
    """Test tensorflow hub encoding DataArray containing image like data."""
    num_samples = 5  # multiple batches, smaller last

    def setUp(self):
        """Set up the test."""
        data = np.random.rand(self.num_samples, 512, 512, 3)
        samples = range(self.num_samples)
        self.data_array = xr.DataArray(
            data,
            dims=['sample', 'height', 'width', 'channel'],
            coords={'sample': list(map(str, samples)),
                    'matched': ('sample', list(map(coord_mapping, samples)))}
        )

    def test_hubencoder(self):
        """Test HubEncoder."""
        nasnet_hub_url = (
            'https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1'
        )
        num_features = 1056
        encoded_image_size = (224, 224)

        encoder = HubEncoder(
            nasnet_hub_url, ms1_encoder='encoding random tensor',
            mudule_name='nasnet'
        )
        encoded_array = encoder(self.data_array)

        self.assertEqual(
            encoded_array.sizes,
            {'sample': self.num_samples, 'hub_feature': num_features}
        )
        self.assertEqual(
            encoded_array.attrs,
            {
                'ms1_encoder': 'encoding random tensor',
                'mudule_name': 'nasnet',
                'encoded_image_size': encoded_image_size
            }
        )


class HubEncoderGreyTestCase(unittest.TestCase):
    """Test tensorflow hub encoding DataArray containing image like data."""
    num_samples = 5  # multiple batches, smaller last

    def setUp(self):
        """Set up the test."""
        data = np.random.rand(self.num_samples, 512, 512, 1)
        samples = range(self.num_samples)
        self.data_array = xr.DataArray(
            data,
            dims=['sample', 'height', 'width', 'channel'],
            coords={'sample': list(map(str, samples)),
                    'matched': ('sample', list(map(coord_mapping, samples)))}
        )

    def test_hubencoder(self):
        """Test HubEncoder."""
        nasnet_hub_url = (
            'https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1'
        )
        num_features = 1056
        encoded_image_size = (224, 224)

        encoder = HubEncoder(
            nasnet_hub_url, ms1_encoder='encoding random tensor',
            mudule_name='nasnet'
        )
        encoded_array = encoder(self.data_array)

        self.assertEqual(
            encoded_array.sizes,
            {'sample': self.num_samples, 'hub_feature': num_features}
        )
        self.assertEqual(
            encoded_array.attrs,
            {
                'ms1_encoder': 'encoding random tensor',
                'mudule_name': 'nasnet',
                'encoded_image_size': encoded_image_size
            }
        )


class FlattenTestCase(unittest.TestCase):
    """Test flattening encoder."""

    height = 10
    width = 10
    channel = 3
    sample = 100

    def setUp(self):
        """Set up the test for an encoder."""
        sample_coordinates = list(map(
            lambda number: f'sample_{number}',
            range(self.sample)
        ))
        self.data_array = xr.DataArray(
            np.random.rand(
                self.sample, self.height,
                self.width, self.channel
            ),
            dims=[
                'sample', 'height',
                'width', 'channel'
            ],
            coords={'sample': sample_coordinates}
        )

    def test_flatten_defaults(self):
        """Test the flatten encoder with default settings."""
        flattener = Flatten(a_flattener='testing_encoder')
        data_array = flattener(self.data_array)
        self.assertEqual(
            list(data_array.sizes.items()),
            [
                ('sample', self.sample),
                ('features', self.height * self.width * self.channel)
            ]
        )
        self.assertEqual(
            data_array.attrs,
            {
                'a_flattener': 'testing_encoder'
            }
        )

    def test_flatten_custom(self):
        """Test the flatten encoder keeping given dimensions."""
        flattener = Flatten(
            dim='variables',
            dim_to_keep='sample',
            a_flattener='testing_encoder'
        )
        data_array = flattener(self.data_array)
        self.assertEqual(
            list(data_array.sizes.items()),
            [
                ('sample', self.sample),
                ('variables', self.height * self.width * self.channel)
            ]
        )
        self.assertEqual(
            data_array.attrs,
            {
                'a_flattener': 'testing_encoder'
            }
        )
        flattener = Flatten(
            dim='variables',
            dim_to_keep='width',
            a_flattener='testing_encoder'
        )
        data_array = flattener(self.data_array)
        self.assertEqual(
            list(data_array.sizes.items()),
            [
                ('width', self.width),
                ('variables', self.height * self.sample * self.channel)
            ]
        )
        self.assertEqual(
            data_array.attrs,
            {
                'a_flattener': 'testing_encoder'
            }
        )
