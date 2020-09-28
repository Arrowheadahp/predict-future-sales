import pandas as pd
import numpy as np

own = pd.read_csv('sub_cat_5.csv')
best = pd.read_csv('sub_gbm_4.csv')

alpha = 0.1

new = own*(1-alpha) + best*alpha

new.to_csv('stack_3.csv', index=False)
print(new.head())
