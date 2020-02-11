"""Components for data branching."""
from .core import Component


class Brancher(Component):
    """An abstract implementation of an brancher class."""

    def __init__(self, attributes={}):
        """
        Initialize the brancher.

        Args:
            attributes (dict): attributes to add to each xr.DataArray
                contained in the resulting iterable of xr.DataArrays.
        """
        self.attributes = attributes

    def __call__(self, an_object):
        """
        Branch an object in an iterable of xr.DataArrays.

        Args:
            an_object (object): an object.

        Returns:
            an iterable of xr.DataArrays.
        """
        raise NotImplementedError
