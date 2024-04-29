import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('train.csv')
titanic_df.head()

titanic_df.info()

titanic_df.describe()

sns.catplot(x='Sex', kind='count', data=titanic_df, orient='h')

plt.show()