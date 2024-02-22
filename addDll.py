from ctypes import *

def add_numbers(a, b):
    return a + b

dll = CDLL(None)


dll.add_numbers.argtypes = [c_int , c_int]
dll.add_numbers.restypes = c_int


