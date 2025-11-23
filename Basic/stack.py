
# Create stack
a_stack = []
print("Stack with type =", type(a_stack))

# Fill stack with append
a_stack.append(3)
a_stack.append(2)
a_stack.append('a')

# Output stack
print(a_stack)

# Read stack
print("Lifo: first value")
print(a_stack.pop())
print("Lifo: second value")
print(a_stack.pop())
print("Lifo: third value")
print(a_stack.pop())

# Stack Test
# print("Lifo: Output with empty stack")
# print(a_stack.pop())
print()




# Load deque
from collections import deque

# Create stack
a_stack = deque()
print("Stack with type =", type(a_stack))

# Fill stack with append
a_stack.append(3)
a_stack.append(2)
a_stack.append('a')

# Output stack
print(a_stack)

# Read stack
print("Lifo: first value")
print(a_stack.pop())
print("Lifo: second value")
print(a_stack.pop())
print("Lifo: third value")
print(a_stack.pop())

# Stack Test
# print("Lifo: Output with empty stack")
# print(a_stack.pop())
print()




# Load Lifoqueue
from queue import LifoQueue
# Create stack
a_stack = LifoQueue()
print("Stack with type =", type(a_stack))

# Fill stack with put
a_stack.put(3)
a_stack.put(2)
a_stack.put('a')

# WARNING: LifoQueue has no direct way to view all elements,
# therefore we must remove them when outputting (with get)

# Read stack
print("Lifo: first value")
print(a_stack.get())
print("Lifo: second value")
print(a_stack.get())
print("Lifo: third value")
print(a_stack.get())

# Stack Test
print("Lifo: Output with empty stack")
# print(a_stack.get()) must not be executed here
# as it would result in an infinite wait loop
print("If the stack is empty, you would wait forever for output.")

# Bypass the waiting time
print("Output without waiting time")
# a_stack.get_nowait()