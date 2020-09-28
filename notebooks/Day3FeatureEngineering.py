#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from itertools import product
from tqdm.notebook import tqdm
import calendar


# In[2]:


DATA_FOLDER = '../Data/'

sales           = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
# item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
# shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv.gz'))
# samplesub       = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv.gz'))


# In[3]:


sales['item_category_id'] = sales.item_id.map(items.item_category_id)
sales.item_cnt_day = sales.item_cnt_day.clip(0, 20)
sales = sales[sales.item_price < 30000]


# In[4]:


# index_cols = ['date_block_num', 'shop_id', 'item_id']

# train = []
# block = sales.date_block_num.unique()

# for shop in tqdm(sales.shop_id.unique()):
#     cur_items = sales.loc[sales.shop_id == shop, 'item_id'].unique()
#     train.append(
#         np.array(list(
#             product(*[block, [shop], cur_items])
#         ))
#     )
    
# train = pd.DataFrame(np.vstack(train), columns=index_cols, dtype=np.int32)


# In[5]:


index_cols = ['date_block_num', 'shop_id', 'item_id']

train = []
# block = sales.date_block_num.unique()

for block in tqdm(sales.date_block_num.unique()):
    cur_items = sales.loc[sales.date_block_num == block, 'item_id'].unique()
    cur_shops = sales.loc[sales.date_block_num == block, 'shop_id'].unique()
    train.append(
        np.array(list(
            product(*[[block], cur_shops, cur_items])
        ))
    )
    
train = pd.DataFrame(np.vstack(train), columns=index_cols, dtype=np.int32)


# In[6]:


test['date_block_num'] = np.full(test.shape[0], 34)


# In[7]:


group = sales.groupby(index_cols).agg({'item_cnt_day': 'sum'})
group.columns = ['item_cnt_month']
group.reset_index(inplace= True)
train = pd.merge(train, group, on= index_cols, how= 'left')
train.item_cnt_month = train.item_cnt_month.fillna(0).clip(0, 20)


# In[8]:


# train = train.sample(3000000)


# In[9]:


length_train = train.shape[0]
length_train


# In[10]:


train = train.append(test.drop(columns=['ID']), ignore_index=True)


# In[11]:


train.shape


# In[12]:


y = train.item_cnt_month[:length_train]
# train.drop(columns=['item_cnt_month'], inplace=True)


# In[13]:


train.head()


# In[14]:


def num_days(block):
    y = int(2013 + block/12)
    m = 1 + block % 12
    return max(calendar.monthcalendar(y, m)[-1])

def wday(block, d):
    y = int(2013 + block/12)
    m = 1 + block % 12
    cal = calendar.monthcalendar(y, m)
    return sum([1 for w in cal if w[d]])


# In[15]:


def lag_feature(df, lags, col='item_cnt_month'):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in tqdm(lags):
        name = col+'_lag_'+str(i)
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', name]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        df[name] = df[name].astype('float16')
    return df


# In[16]:


def add(train, index_cols, name, col='item_cnt_month', func='mean'):
    group = train[train.date_block_num.isin(sales.date_block_num.unique())].groupby(index_cols).agg({col: func})
    group.columns = [name]
    group.reset_index(inplace= True)
    train = pd.merge(train, group, on= index_cols, how= 'left')
    train[name] = train[name].fillna(0).astype('float16')
    return train


# In[17]:


train['year'] = (train.date_block_num/12 + 2013).astype(int).astype('int32')
train['month'] = (train.date_block_num % 12 + 1).astype(int).astype('int32')

sales['month'] = (sales.date_block_num % 12 + 1).astype(int).astype('int32')


# In[18]:


train['num_days'] = train.date_block_num.map(pd.Series([num_days(i) for i in range(40)])).astype('int32')

train['num_sat'] = train.date_block_num.map(pd.Series([wday(i, 5) for i in range(40)])).astype('int32')
# for d, wk in enumerate(['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']):
#     train['num_'+ wk] = train.date_block_num.map(pd.Series([wday(i, d) for i in range(40)]))


# In[19]:


prices       = sales.groupby('item_id' ).item_price.mean()
prices_month = sales.groupby('month'   ).item_price.mean()
prices_shop  = sales.groupby('shop_id' ).item_price.mean()

train['item_category_id'] = train.item_id.map(items.item_category_id).astype('int32')
train['price'] = train.item_id.map(prices).astype('float16')
train['price_month'] = train.month.map(prices_month).astype('float16')
train['price_shop'] = train.shop_id.map(prices_shop).astype('float16')


# ## target encoding

# In[20]:


# [train.date_block_num.isin(sales.date_block_num.unique())]
block_sales = sales.date_block_num.unique()

count_month = train[:length_train].groupby('month'  ).item_cnt_month.mean()
count_shop  = train[:length_train].groupby('shop_id').item_cnt_month.mean()
count_item  = train[:length_train].groupby('item_id').item_cnt_month.mean()

train['cnt_month'] = train.month.map(count_month).astype('float16')
train['cnt_shop'] = train.shop_id.map(count_shop).astype('float16')
train['cnt_item'] = train.item_id.map(count_item).astype('float16')


# In[21]:


# train = add(train, ['date_block_num', 'item_id'], 'avg_cnt_month-item')
# train = add(train, ['date_block_num', 'shop_id'], 'avg_cnt_month-shop')
train = add(train, ['item_id', 'shop_id'], 'avg_cnt_item-shop')

train = add(train, ['month', 'item_id'], 'avg_cnt_month')
train = add(train, ['month', 'shop_id'], 'avg_cnt_shop')


# In[22]:


## lag
# for col in [
#     'item_cnt_month', 
#     'cnt_shop', 
#     'cnt_month', 
#     'cnt_item', 
#     'avg_cnt_month-item', 
#     'avg_cnt_month-shop', 
#     'avg_cnt_item-shop'
# ]:
#     train = lag_feature(train, [1, 2, 3, 12, 24], 'item_cnt_month')


# ## impute

# In[23]:


train.price = train.price.fillna(train.groupby('item_category_id').price.transform('mean'))
train.cnt_item = train.cnt_item.fillna(train.groupby('item_category_id').cnt_item.transform('mean'))


# In[24]:


ZD = train.copy()
##ZD = pd.get_dummies(ZD, columns= ['year', 'month', 'num_days', 'num_sat', 'shop_id', 'item_category_id'])


# In[25]:


test = ZD[length_train:]
train = ZD[:length_train]


# In[26]:


def information(rtd):
    info = pd.DataFrame({'Column': rtd.columns})
    info['nunique'] = rtd.nunique().values
    info['dtypes']  = rtd.dtypes.values
    info['isNull']  = rtd.isna().sum().values
    info['mode']    = rtd.mode().values[0]
    return info.set_index('Column')


# In[27]:


train.to_hdf('../Dump/train2.h5', 'train', 'w')
test.to_hdf('../Dump/test2.h5', 'test', 'w')


# In[28]:


information(test)


# In[29]:


##train.info()


# In[ ]:




