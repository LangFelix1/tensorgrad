import numpy as np
import cupy as cp

_current_backend = np

def use_cpu():
    global _current_backend
    _current_backend = np

def use_gpu():
    global _current_backend
    _current_backend = cp

def backend():
    return _current_backend