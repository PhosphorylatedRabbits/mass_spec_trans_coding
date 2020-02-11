"""Tests for data branchers."""
import os
import shutil
import tempfile
import unittest

import imageio
import numpy as np
import xarray as xr

from .encoder import Flatten
from .io import PNGReader
from .operation import Broadcast, Compose, ZipMap


class ComposeTestCase(unittest.TestCase):
    """Test composition encoder."""

    height = 10
    width = 10
    channel = 3
    sample = 100

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

    def test_composition(self):
        """Test the composition to define a pipeline."""
        composition = Compose(
            [
                PNGReader(a_reader='testing_reader'),
                Flatten(a_flattener='testing_flattener')
            ],
            a_composition='testing_composition'
        )
        data_array = composition(self.globbing_pattern)
        self.assertEqual(
            list(data_array.sizes.items()),
            [
                ('sample', self.sample),
                ('features', self.height * self.width * self.channel)
            ]
        )
        self.assertEqual(
            dict(data_array.attrs),
            {
                'a_reader': 'testing_reader',
                'a_flattener': 'testing_flattener',
                'a_composition': 'testing_composition'
            }
        )

    def tearDown(self):
        """Delete .png images and the temporary folder."""
        shutil.rmtree(self.directory)


class BroadcastTestCase(unittest.TestCase):
    """Test broadcast brancher."""

    height = 10
    width = 10
    channel = 3
    sample = 100

    def setUp(self):
        """Set up the test for a brancher."""
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

    def test_broadcast(self):
        """Test the broadcast brancher."""
        broadcaster = Broadcast(
            [
                Flatten(
                    dim_to_keep='sample',
                    a_flattener='testing_encoder'
                ),
                Flatten(
                    dim='variables',
                    dim_to_keep='width',
                    a_flattener='testing_encoder'
                )
            ],
            a_broadcast='testing_broadcast'
        )
        data_arrays = list(broadcaster(self.data_array))
        self.assertEqual(
            list(data_arrays[0].sizes.items()),
            [
                ('sample', self.sample),
                ('features', self.height * self.width * self.channel)
            ]
        )
        self.assertEqual(
            dict(data_arrays[0].attrs),
            {
                'a_flattener': 'testing_encoder',
                'a_broadcast': 'testing_broadcast'
            }
        )
        self.assertEqual(
            list(data_arrays[1].sizes.items()),
            [
                ('width', self.width),
                ('variables', self.height * self.sample * self.channel)
            ]
        )
        self.assertEqual(
            dict(data_arrays[1].attrs),
            {
                'a_flattener': 'testing_encoder',
                'a_broadcast': 'testing_broadcast'
            }
        )


class ZipMapTestCase(unittest.TestCase):
    """Test map encoder."""

    height = 10
    width = 10
    channel = 3
    sample = 100
    number_of_arrays = 2

    def setUp(self):
        """Set up a list of arrays with a common coordinate."""
        sample_coordinates = list(map(
            lambda number: f'sample_{number}',
            range(self.sample)
        ))
        self.data_arrays = [
            xr.DataArray(
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
            for _ in range(self.number_of_arrays)
        ]

    def test_zipmap(self):
        """Test the zipmap operation."""
        zip_mapper = ZipMap(
            [
                Flatten(
                    dim_to_keep='sample',
                    a_flattener='testing_encoder'
                ),
                Flatten(
                    dim='variables',
                    dim_to_keep='width',
                    a_flattener='testing_encoder'
                )
            ],
            a_zipmap='testing_zipmap'
        )
        data_arrays = list(zip_mapper(self.data_arrays))
        self.assertEqual(
            list(data_arrays[0].sizes.items()),
            [
                ('sample', self.sample),
                ('features', self.height * self.width * self.channel)
            ]
        )
        self.assertEqual(
            dict(data_arrays[0].attrs),
            {
                'a_flattener': 'testing_encoder',
                'a_zipmap': 'testing_zipmap'
            }
        )
        self.assertEqual(
            list(data_arrays[1].sizes.items()),
            [
                ('width', self.width),
                ('variables', self.height * self.sample * self.channel)
            ]
        )
        self.assertEqual(
            dict(data_arrays[1].attrs),
            {
                'a_flattener': 'testing_encoder',
                'a_zipmap': 'testing_zipmap'
            }
        )
