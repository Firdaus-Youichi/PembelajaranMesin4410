import re
import string
import time
from copy import deepcopy
import pandas as pd
df = pd.read_csv('Dataset_Sentimen_Emosi.csv')
df.head()
df = df.drop(['Emosi'], axis=1)
df
df.info()