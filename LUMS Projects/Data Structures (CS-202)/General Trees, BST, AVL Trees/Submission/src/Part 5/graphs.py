import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to fit for BST (assuming worst case O(n))
def bst_fit_func(x, a, b):
    return a * x + b

# Function to fit for AVL (assuming typical case O(log n))
def avl_fit_func(x, a, b):
    return a * np.log(x) + b

# Read the data
bst_data = pd.read_csv('BST_InsertionData.csv')
avl_data = pd.read_csv('AVL_InsertionData.csv')

# Separate the data into X (number of elements) and Y (time)
x_bst = bst_data['Elements']
y_bst = bst_data['Time(us)']
x_avl = avl_data['Elements']
y_avl = avl_data['Time(us)']

# Curve fitting
bst_params, _ = curve_fit(bst_fit_func, x_bst, y_bst)
avl_params, _ = curve_fit(avl_fit_func, x_avl, y_avl)

# Plot the data
plt.figure(figsize=(12, 8))

# Plot BST data and fit
plt.scatter(x_bst, y_bst, label='BST Data', alpha=0.5)
plt.plot(x_bst, bst_fit_func(x_bst, *bst_params), color='red')

# Plot AVL data and fit
plt.scatter(x_avl, y_avl, label='AVL Data', alpha=0.5)
plt.plot(x_avl, avl_fit_func(x_avl, *avl_params), color='black')

plt.xlabel('Number of Elements')
plt.ylabel('Time (microseconds)')
plt.title('Insertion Time Comparison b/w BST and AVL Trees')
plt.legend()
plt.savefig('BST_VS_AVL.png')

print('\nGraphs saved successfully')