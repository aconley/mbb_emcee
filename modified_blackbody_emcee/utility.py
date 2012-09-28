__all__ = ["isiterable"]


def isiterable(obj):
    """Returns `True` if the given object is iterable."""
    import collections, numpy

    # Numpy arrays are in collections.Iterable no matter what, but if you
    # attempt to iterate over a 0-d array, it throws a TypeError.
    if isinstance(obj, numpy.ndarray) and len(obj.shape) == 0:
        return False

    if isinstance(obj, collections.Iterable):
        return True

    try:
        iter(obj)
        return True
    except TypeError:
        return False
