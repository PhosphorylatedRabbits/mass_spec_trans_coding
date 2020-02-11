"""Abstract implementation of components and operations working on them."""


class Component(object):
    """An abstract component implementation."""
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
        A abstract component implementation

        Args:
            an_object (object): input for the component.

        Returns:
            an object processed by the component.
        """
        raise NotImplementedError


class Operation(Component):
    """An abstract implementation of a higher order class
    to define pipelines."""
    def __init__(self, attributes={}):
        """
        Initialize an operation.

        Args:
            kwargs (dict): arguments to pass to Brancher as attributes.
        """
        super(Operation, self).__init__(attributes)

    def __call__(self, an_object):
        """
        A abstract operation implementation

        Args:
            an_object (object): input for the operation.

        Returns:
            an object processed by the operation.
        """
        raise NotImplementedError


class SingleOperation(Operation):
    """An abstract implementation of an operation with a single component."""
    def __init__(self, component, **kwargs):
        """
        Initialize an operation where a single component is applied.

        Args:
            component (Component): a component.
            kwargs (dict): arguments to pass to Brancher as attributes.
        """
        super(SingleOperation, self).__init__(**kwargs)
        self.component = component


class MultipleOperation(Operation):
    """An abstract implementation of an operation with multiple components."""
    def __init__(self, components, **kwargs):
        """
        Initialize an operation where multiple components are applied.

        Args:
            components (iterable): an iterable containing components.
            kwargs (dict): arguments to pass to Brancher as attributes.
        """
        super(MultipleOperation, self).__init__(**kwargs)
        self.components = components
