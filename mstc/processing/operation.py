"""Higher order operations initialized with a component or components."""
from functools import reduce

from .core import MultipleOperation, SingleOperation


class Compose(MultipleOperation):
    """Implement a pipeline to execute a sequence of components.
    Propagating attributes."""

    def __init__(self, components, **kwargs):
        """
        Initialize a pipeline.

        Args:
            components (iterable): an iterable containing components.
            kwargs (dict): arguments to pass to Encoder as attributes.
        """
        super(Compose, self).__init__(
            components=components, attributes=kwargs
        )

    def __call__(self, an_object):
        """
        Execute a composition of components.

        Args:
            an_object (object): an input for the composition.

        Returns:
            a xr.DataArray or iterable of xr.DataArray generated from the
            composition.
        """
        attributes = self.attributes  # propagate attributes
        for component in self.components:
            component.attributes.update(attributes)
            an_object = component(an_object)
            attributes.update(component.attributes)  # after call

        return an_object


class Broadcast(MultipleOperation):
    """Broadcast an input using multiple components."""

    def __init__(self, components, **kwargs):
        """
        Initialize the operation.

        Args:
            components (iterable): an iterable containing components.
            kwargs (dict): arguments to pass to Brancher as attributes.
        """
        super(Broadcast, self).__init__(
            components=components, attributes=kwargs
        )

    def __call__(self, an_object):
        """
        Broadcast an object into an iterable of xr.DataArrays
        using multiple components.

        Args:
            an_object (object): an object.

        Returns:
            an iterable of xr.DataArrays.
        """
        operation_attributes = self.attributes.copy()  # collect attributes
        for component in self.components:
            self.attributes.update(component.attributes)
            component.attributes.update(operation_attributes)
            yield component(an_object)


class BroadcastMap(MultipleOperation):
    """Broadcast input to multiple Map components initialized on the fly."""
    def __init__(self, components, **kwargs):
        """
        Initialize the operation.

        Args:
            components (iterable): an iterable containing components.
            kwargs (dict): arguments to pass to Brancher as attributes.
        """
        super(BroadcastMap, self).__init__(
            components=components, attributes=kwargs
        )

    def __call__(self, an_iterable):
        """
        Apply each object to all components.

        Args:
            an_iterable (iterable): an iterable of objects.

        Returns:
            an iterable of xr.DataArrays.

        """
        an_iterable = list(an_iterable)
        operation_attributes = self.attributes.copy()  # collect attributes
        for component in self.components:
            self.attributes.update(component.attributes)
            component.attributes.update(operation_attributes)
            for an_object in Map(component, **operation_attributes)(an_iterable):  # noqa
                yield an_object


class ZipMap(MultipleOperation):
    """Map component of an iterable to respective xr.DataArray of an iterable.
    """

    def __init__(self, components, **kwargs):
        """
        Initialize the zip.

        Args:
            components (iterable): an iterable containing components.
            kwargs (dict): arguments to pass to components as attributes.
        """
        super(ZipMap, self).__init__(
            components=components, attributes=kwargs
        )

    def __call__(self, an_iterable):
        """
        Encoding an iterable to an iterable of xr.DataArrays, the attributes
        are added to xr.DataArray.

        Args:
            an_iterable (iterable): an_iterable.
        Returns:
            an iterable of xr.DataArrays.
        """
        operation_attributes = self.attributes.copy()  # collect attributes
        for component, an_object in zip(self.components, an_iterable):
            self.attributes.update(component.attributes)
            component.attributes.update(operation_attributes)
            yield component(an_object)


class Map(SingleOperation):
    """Apply component to all objects in iterable"""
    def __init__(self, component, **kwargs):
        """
        Initialize the reduction.

        Args:
            component (Component): a component accepting iterable.
            kwargs (dict): arguments to pass to component as attributes.
        """
        super(Map, self).__init__(
            component=component, attributes=kwargs
        )

    def __call__(self, an_iterable):
        """
        Map the component over an iterable using the standard map.

        Args:
            an_iterable (iterable): an iterable of objects.

        Returns:
            a map object (iterable).
        """
        return map(self.component.__call__, an_iterable)


class Reduce(SingleOperation):
    """Apply component with iterable input with single returned object."""
    def __init__(self, component, **kwargs):
        """
        Initialize the reduction.

        Args:
            component (Component): a component accepting iterable.
            kwargs (dict): arguments to pass to component as attributes.
        """
        super(Reduce, self).__init__(
            component=component, attributes=kwargs
        )

    def __call__(self, an_iterable):
        """
        Reduce an iterable to a single object using the standard reduce.

        Args:
            an_iterable (iterable): an iterable of objects.

        Returns:
            an object.
        """
        return reduce(self.component.__call__, an_iterable)
