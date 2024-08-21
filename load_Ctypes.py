from ctypes import *

# Load the shared library (replace './mylib.so' with './mylib.dll' on Windows)
dll = CDLL('./mylib.so')

# Define the argument types and return type
dll.add_number.argtypes = [c_int, c_int]
dll.add_number.restype = c_int

# Call the C function
result = dll.add_number(10, 20)

print("Result from add_number:", result)

