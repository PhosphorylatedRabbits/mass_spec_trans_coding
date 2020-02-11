"""Components for data I/O."""
import glob
import xarray as xr
from dask.array.image import imread as lazy_imread
from .core import Component


class Reader(Component):
    """An abstract implementation of a reader class."""

    def __init__(self, attributes={}):
        """
        Initialize the reader.

        Args:
            attributes (dict): attributes to add to the resulting
                xr.DataArray.
        """
        self.attributes = attributes

    def __call__(self, globbing_pattern):
        """
        Parse samples from a globbing pattern and generate an xr.DataArray.

        Args:
            globbing_pattern (str): a globbing pattern.

        Returns:
            a xr.DataArray.
        """
        raise NotImplementedError


class PNGReader(Reader):
    """A .png reader."""

    def __init__(self, **kwargs):
        """
        Initialize the .png reader.

        Args:
            kwargs (dict): arguments to pass to Reader as attributes.
        """
        super(PNGReader, self).__init__(attributes=kwargs)

    def __call__(self, globbing_pattern):
        """
        Parse samples from a globbing pattern for png files
        and generate an xr.DataArray using lazy loading.

        Args:
            globbing_pattern (str): a globbing pattern.

        Returns:
            a xr.DataArray (sample, height, width, channels) assembled
            stacking the images and adding the filepath as a coordinate
            on the sample dimension.
        """
        try:
            data_array = xr.DataArray(
                lazy_imread(globbing_pattern),
                dims=['sample', 'height', 'width', 'channel']
            )
        except ValueError:  # Grayscale image
            data_array = xr.DataArray(
                lazy_imread(globbing_pattern),
                dims=['sample', 'height', 'width']
            ).expand_dims(dim='channel', axis=-1)
        return data_array.assign_coords(
                sample=sorted(glob.glob(globbing_pattern))
            ).assign_attrs(
                self.attributes
            )
