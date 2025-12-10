import numpy
import matplotlib.pyplot as plt
import pickle
import timeit
import os

plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

def sizeof_fmt(num, suffix='B'):
    '''Human-readable file size.'''
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}P{suffix}'

def save(filename: str, data):
    '''
    Save a dictionary (with arbitrary Python objects) to a binary file.

    Parameters
    ----------
    filename : str
        The filename to save the dictionary to (should end in .pkl).
    data : dict
        The dictionary to save.
    '''
    t0 = timeit.default_timer()
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    t1 = timeit.default_timer()
    
    size = os.path.getsize(filename)        # bytes
    size_MB = size / (1024 * 1024)          # convert to MB
    dt = t1 - t0

    print(f'[pyjack.save] Saved data to {filename}, with size {sizeof_fmt(size)} ({size_MB/dt:.3f} MB/s)')


def load(filename: str):
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
    size = os.path.getsize(filename)        # bytes
    size_MB = size / (1024 * 1024)

    t0 = timeit.default_timer()
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    t1 = timeit.default_timer()

    dt = t1 - t0

    print(f'[pyjack.load] Loaded data from {filename}, with size {sizeof_fmt(size)} ({size_MB/dt:.3f} MB/s)')

    if hasattr(obj, 'creation_info'):
        info = obj.creation_info
        print(f"[pyjack.load] Created on: {info.get('timestamp')} by {info.get('hostname')} ({info.get('ip')})")

    return obj

def plt_errorbar_fill_color(x,y,dy,color='C0',label=None,alpha=0.3):
    '''Plot errorbar and fill it with color
    
    Plot (x,y) labeled as label, adding dashed lines (x,y-dy) and (x,y+dy)
    and filling between coloring with color faded with alpha'''
    if label:
        plt.plot(x,y,color=color,label=label)
    else:
        plt.plot(x,y,color=color)
    plt.plot(x,y-dy,color=color,linestyle='--')
    plt.plot(x,y+dy,color=color,linestyle='--')
    plt.fill_between(x,y-dy,y+dy,color=color,alpha=alpha)

def plt_errorbar_fillx_color(y,x,dx,color='C0',label=None,alpha=0.3):
    '''Plot errorbar and fill it with color along x
    
    Plot vertical line at x labeled as label, adding dashed lines (x-dx,y) and (x+dx,y)
    and filling between coloring with color faded with alpha'''
    if label:
        plt.axvline(x,color=color,label=label)
    else:
        plt.axvline(x,color=color)
    plt.axvline(x-dx,color=color,linestyle='--')
    plt.axvline(x+dx,color=color,linestyle='--')
    plt.fill_betweenx(y,x-dx,x+dx,color=color,alpha=alpha)