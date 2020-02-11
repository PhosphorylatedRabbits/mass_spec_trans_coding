"""Components for data aggreation."""
import xarray as xr
from .core import Component


class Aggregator(Component):
    """An abstract implementation of an aggregator class."""

    def __init__(self, attributes={}):
        """
        Initialize the aggregator.

        Args:
            attributes (dict): attributes to add to the resulting
                xr.DataArray.
        """
        self.attributes = attributes

    def __call__(self, data_arrays):
        """
        Aggregate an iterable of xr.DataArrays in a single xr.DataArray.

        Args:
            data_arrays (iterable): an iterable containing xr.DataArrays.

        Returns:
            a xr.DataArray.
        """
        raise NotImplementedError


class Stacker(Aggregator):
    """An implementation for a stack aggregator."""

    def __init__(self, dim='new', axis=0, **kwargs):
        """
        Initialize the stack aggregator.

        Args:
            dim (str): name of the new dimension, defaults to 'new'.
            axis (int): axis where to insert the dimension, defaults to 0.
            kwargs (dict): arguments to pass to Aggregator as attributes.
        """
        super(Stacker, self).__init__(attributes=kwargs)
        self.dim = dim
        self.axis = axis

    def __call__(self, data_arrays):
        """
        Aggregate an iterable of xr.DataArrays in a single xr.DataArray
        by stacking on a new dimension.

        Args:
            data_arrays (iterable): an iterable containing xr.DataArrays.

        Returns:
            a xr.DataArray.
        """
        return xr.concat(
            map(
                lambda data_array: data_array.expand_dims(
                    dim=self.dim, axis=self.axis
                ),
                data_arrays
            ),
            dim=self.dim
        ).assign_attrs(self.attributes)
