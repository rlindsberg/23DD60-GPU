import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

df = pd.read_csv('exp1.csv')
array_n = np.array(df['N'])
array_tbp = np.array(df['TBP'])
array_n_iter = np.array(df['N_ITER'])
array_time = np.array(df['TIME'])
array_pi = np.array(df['PI'])

plt.scatter(array_n_iter, array_time)
plt.ylabel('GPU time/us')
plt.xlabel('Number of iterations per thread')
# plt.legend()
plt.show()

df = pd.read_csv('exp3.csv')
array_n = np.array(df['N'])
array_tbp = np.array(df['TBP'])
array_n_iter = np.array(df['N_ITER'])
array_time = np.array(df['TIME'])
array_pi = np.array(df['PI'])

plt.scatter(array_tbp, array_time)
plt.ylabel('GPU time/us')
plt.xlabel('Number of threads per block')
# plt.legend()
plt.show()

df = pd.read_csv('exp2.csv')
array_n = np.array(df['N'])
array_tbp = np.array(df['TBP'])
array_n_iter = np.array(df['N_ITER'])
array_time = np.array(df['TIME'])
array_pi = np.array(df['PI'])
array_n_blocks = array_n / array_tbp

plt.scatter(array_n_blocks, array_time)
plt.ylabel('GPU time/us')
plt.xlabel('Number of blocks')
# plt.legend()
plt.show()