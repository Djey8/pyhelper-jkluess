import numpy as np

# Type Array
array = np.array([])
print("Type:", type(array))

# 5-dim.
print("5-dim. Array a:")
a = np.array([1,2,3,4,5])
print(a)

# 2x5-dim. Array
print("2x5-dim. Array b:")
b = np.array([[1,2,3,4,5], [0,1,0,1,0]])
print(b)
print("Loop: increase each value in b by 1:")
for element in b:
    element+=1
    print(element)
    
    
# An operation possible with arrays
# https://numpy.org/doc/stable/reference/routines.array-manipulation.html
print("a/2:")
print(a/2)

print("transpose:")
print(b.transpose())

print("shape:")
print(np.shape(b))

print("flip:")
print(np.flip(b))