import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('exp1.csv')

sns.set(rc={'figure.figsize': (12, 8)})

sns.catplot(x="matrix_size", y="time", hue="device_n_lib", data=df, kind="bar")
plt.yscale("log")

plt.savefig('plot.png')
plt.show()
