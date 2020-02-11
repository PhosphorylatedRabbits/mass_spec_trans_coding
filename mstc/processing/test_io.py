"""Tests for data I/O."""
import os
import shutil
import imageio
import tempfile
import unittest
import numpy as np
from .io import PNGReader


def silence_imageio_warning(*args, **kwargs):
    """patch image conversion warning"""
    pass


imageio.core.util._precision_warn = silence_imageio_warning


class PNGReaderRGBTestCase(unittest.TestCase):
    """Test reading of rgb images in xr.DataArray."""

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

    def test_pngreader(self):
        """Test the .png reader."""
        reader = PNGReader(a_reader='testing_reader')
        data_array = reader(self.globbing_pattern)
        self.assertEqual(
            dict(data_array.sizes),
            dict([
                ('sample', self.sample),
                ('height', self.height),
                ('width', self.width),
                ('channel', self.channel)
            ])
        )
        self.assertEqual(
            data_array.attrs,
            {
                'a_reader': 'testing_reader'
            }
        )

    def tearDown(self):
        """Delete .png images and the temporary folder."""
        shutil.rmtree(self.directory)


class PNGReaderGrayTestCase(unittest.TestCase):
    """Test reading of grayscale images in xr.DataArray."""

    height = 10
    width = 10
    sample = 100

    def setUp(self):
        """Generate .png images in a temporary folder."""
        self.directory = tempfile.mkdtemp()
        for i in range(self.sample):
            imageio.imwrite(
                os.path.join(self.directory, f'{str(i).zfill(3)}.png'),
                np.random.rand(self.height, self.width)
            )
        self.globbing_pattern = os.path.join(
            self.directory, '*.png'
        )

    def test_pngreader(self):
        """Test the .png reader."""
        reader = PNGReader(a_reader='testing_reader')
        data_array = reader(self.globbing_pattern)
        self.assertEqual(
            dict(data_array.sizes),
            dict([
                ('sample', self.sample),
                ('height', self.height),
                ('width', self.width),
                ('channel', 1)
            ])
        )
        self.assertEqual(
            data_array.attrs,
            {
                'a_reader': 'testing_reader'
            }
        )

    def tearDown(self):
        """Delete .png images and the temporary folder."""
        shutil.rmtree(self.directory)
