import numpy
import pickle

def pretty_print(x, dx, further_digits=1):
    '''
    Return a string representation of x with uncertainty dx.

    Parameters
    ----------
    x : float
        The value to be printed.
    dx : float
        The uncertainty associated to x.
    further_digits : int, optional
        The number of digits to be printed beyond the uncertainty (default: 1).

    Returns
    -------
    string
        A string representation of x with uncertainty dx.
    '''
    if dx == 0:
        return f'{x}(0)'
    decimal_places = max(0, -int(numpy.floor(numpy.log10(dx)))) + further_digits
    return f'{x:.{decimal_places}f}({int(dx * 10**decimal_places)})'

def save(data: dict, filename: str):
    '''
    Save a dictionary (with arbitrary Python objects) to a binary file.

    Parameters
    ----------
    data : dict
        The dictionary to save.
    filename : str
        The filename to save the dictionary to (should end in .pkl).
    '''
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load(filename: str) -> dict:
    '''
    Load a dictionary from a binary file.

    Parameters
    ----------
    filename : str
        The pickle file to load from.

    Returns
    -------
    dict
        The loaded dictionary.
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)
