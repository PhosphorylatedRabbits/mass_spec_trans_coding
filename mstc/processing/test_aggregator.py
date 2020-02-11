"""Tests for data aggregation."""
import unittest

import numpy as np
import xarray as xr

from .aggregator import Stacker


class StackerTestCase(unittest.TestCase):
    """Test the stack aggregator."""

    height = 10
    width = 10
    channel = 3
    sample = 100
    number_of_arrays = 100

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
                coords={'sample': sample_coordinates,
                        'matched': ('sample', sample_coordinates)},
                attrs={'existing': 'pre existing attribute'}
            )
            for _ in range(self.number_of_arrays)
        ]

    def test_stacker_defaults(self):
        """Test the stack aggregator with default settings."""
        stacker = Stacker(an_aggregator='testing_aggregator')
        data_array = stacker(self.data_arrays)
        self.assertEqual(
            list(data_array.sizes.items()),
            [
                ('new', self.number_of_arrays),
                ('sample', self.sample),
                ('height', self.height),
                ('width', self.width),
                ('channel', self.channel)
            ]
        )
        self.assertEqual(
            data_array.attrs,
            {
                'an_aggregator': 'testing_aggregator',
                'existing': 'pre existing attribute'
            }
        )
        self.assertEqual(
            list(data_array.coords.keys()),
            ['sample', 'matched']
        )

    def test_stacker_custom(self):
        """Test the stack aggregator on a specific axis."""
        stacker = Stacker(
            dim='modality',
            axis=4,
            an_aggregator='testing_aggregator'
        )
        data_array = stacker(self.data_arrays)
        self.assertEqual(
            list(data_array.sizes.items()),
            [
                ('sample', self.sample),
                ('height', self.height),
                ('width', self.width),
                ('channel', self.channel),
                ('modality', self.number_of_arrays)
            ]
        )
        self.assertEqual(
            data_array.attrs,
            {
                'an_aggregator': 'testing_aggregator',
                'existing': 'pre existing attribute'
            }
        )

    def test_stacker_different_attr_value(self):
        stacker = Stacker(an_aggregator='testing_aggregator')
        changed_attr_array = self.data_arrays[0].assign_attrs({
            'existing': 'changed attribute'
        })

        data_array = stacker([changed_attr_array, self.data_arrays[1]])
        self.assertEqual(  # not matching attr values are lost
            data_array.attrs,
            {
                'an_aggregator': 'testing_aggregator',
            }
        )
