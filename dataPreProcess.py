import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#loading data and annotating columns
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)

#seaborn dataset visualization

sns.set_theme()
sns.displot(data=data, x="MEDV", kde=True)


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
axs = axs.flatten()

for k,v in data.items():
    sns.distplot(v, ax=axs[index])
    index = index + 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()
#print(data)