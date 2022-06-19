```python
import pandas as pd
import numpy as np
from shutil import copy, rmtree 
import os
from sklearn.model_selection import train_test_split
```


```python
data_pd = pd.read_csv('HAM10000_metadata.csv')
data_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lesion_id</th>
      <th>image_id</th>
      <th>dx</th>
      <th>dx_type</th>
      <th>age</th>
      <th>sex</th>
      <th>localization</th>
      <th>dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HAM_0000118</td>
      <td>ISIC_0027419</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HAM_0000118</td>
      <td>ISIC_0025030</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HAM_0002730</td>
      <td>ISIC_0026769</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HAM_0002730</td>
      <td>ISIC_0025661</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HAM_0001466</td>
      <td>ISIC_0031633</td>
      <td>bkl</td>
      <td>histo</td>
      <td>75.0</td>
      <td>male</td>
      <td>ear</td>
      <td>vidir_modern</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_count = data_pd.groupby('lesion_id').count()
df_count.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>dx</th>
      <th>dx_type</th>
      <th>age</th>
      <th>sex</th>
      <th>localization</th>
      <th>dataset</th>
    </tr>
    <tr>
      <th>lesion_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HAM_0000000</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>HAM_0000001</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HAM_0000002</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>HAM_0000003</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HAM_0000004</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_count = df_count[df_count['dx'] == 1]
df_count.reset_index(inplace=True)
```


```python
def duplicates(x):
    unique = set(df_count['lesion_id'])
    if x in unique:
        return 'no' 
    else:
        return 'duplicates'
```


```python
data_pd['is_duplicate'] = data_pd['lesion_id'].apply(duplicates)
data_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lesion_id</th>
      <th>image_id</th>
      <th>dx</th>
      <th>dx_type</th>
      <th>age</th>
      <th>sex</th>
      <th>localization</th>
      <th>dataset</th>
      <th>is_duplicate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HAM_0000118</td>
      <td>ISIC_0027419</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HAM_0000118</td>
      <td>ISIC_0025030</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HAM_0002730</td>
      <td>ISIC_0026769</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HAM_0002730</td>
      <td>ISIC_0025661</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HAM_0001466</td>
      <td>ISIC_0031633</td>
      <td>bkl</td>
      <td>histo</td>
      <td>75.0</td>
      <td>male</td>
      <td>ear</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_count = data_pd[data_pd['is_duplicate'] == 'no']
```


```python
train, test_df = train_test_split(df_count, test_size=0.15, stratify=df_count['dx'])
```


```python
def identify_trainOrtest(x):
    test_data = set(test_df['image_id'])
    if str(x) in test_data:
        return 'test'
    else:
        return 'train'

#creating train_df
data_pd['train_test_split'] = data_pd['image_id'].apply(identify_trainOrtest)
train_df = data_pd[data_pd['train_test_split'] == 'train']
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lesion_id</th>
      <th>image_id</th>
      <th>dx</th>
      <th>dx_type</th>
      <th>age</th>
      <th>sex</th>
      <th>localization</th>
      <th>dataset</th>
      <th>is_duplicate</th>
      <th>train_test_split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HAM_0000118</td>
      <td>ISIC_0027419</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HAM_0000118</td>
      <td>ISIC_0025030</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HAM_0002730</td>
      <td>ISIC_0026769</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HAM_0002730</td>
      <td>ISIC_0025661</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HAM_0001466</td>
      <td>ISIC_0031633</td>
      <td>bkl</td>
      <td>histo</td>
      <td>75.0</td>
      <td>male</td>
      <td>ear</td>
      <td>vidir_modern</td>
      <td>duplicates</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lesion_id</th>
      <th>image_id</th>
      <th>dx</th>
      <th>dx_type</th>
      <th>age</th>
      <th>sex</th>
      <th>localization</th>
      <th>dataset</th>
      <th>is_duplicate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9206</th>
      <td>HAM_0005250</td>
      <td>ISIC_0029792</td>
      <td>nv</td>
      <td>histo</td>
      <td>40.0</td>
      <td>female</td>
      <td>foot</td>
      <td>rosendahl</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6374</th>
      <td>HAM_0000076</td>
      <td>ISIC_0028605</td>
      <td>nv</td>
      <td>follow_up</td>
      <td>30.0</td>
      <td>female</td>
      <td>trunk</td>
      <td>vidir_molemax</td>
      <td>no</td>
    </tr>
    <tr>
      <th>989</th>
      <td>HAM_0000613</td>
      <td>ISIC_0031253</td>
      <td>bkl</td>
      <td>consensus</td>
      <td>60.0</td>
      <td>male</td>
      <td>hand</td>
      <td>vidir_molemax</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4557</th>
      <td>HAM_0001261</td>
      <td>ISIC_0025332</td>
      <td>nv</td>
      <td>follow_up</td>
      <td>55.0</td>
      <td>female</td>
      <td>lower extremity</td>
      <td>vidir_molemax</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4367</th>
      <td>HAM_0002328</td>
      <td>ISIC_0026896</td>
      <td>nv</td>
      <td>follow_up</td>
      <td>40.0</td>
      <td>female</td>
      <td>back</td>
      <td>vidir_molemax</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(train_df),len(test_df)#(9187, 828)
```




    (9187, 828)




```python
# Image id of train and test images
train_list = list(train_df['image_id'])
test_list = list(test_df['image_id'])

# len(train_list),len(test_list)#(9187, 828)

#Set the image_id as the index in data_pd
data_pd.set_index('image_id', inplace=True)
```


```python
#create store
train_dir = os.path.join(os.getcwd(), 'train_dir')
test_dir = os.path.join(os.getcwd(), 'test_dir')
```


```python
os.mkdir(train_dir)
os.mkdir(test_dir)
```


```python
# Image id of train and test images
train_list = list(train_df['image_id'])
test_list = list(test_df['image_id'])
```


```python
targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
```


```python
for i in targetnames:
    directory1=train_dir+'/'+i
    directory2=test_dir+'/'+i
    os.mkdir(directory1)
    os.mkdir(directory2)
```


```python
for image in train_list:
    file_name = image+'.jpg'
    label = data_pd.loc[image, 'dx']

    # path of source image 
    source = os.path.join('Images', file_name)

    # copying the image from the source to target file
    target = os.path.join(train_dir, label, file_name)

    copy(source, target)
```


```python
for image in test_list:

    file_name = image+'.jpg'
    label = data_pd.loc[image, 'dx']

    # path of source image 
    source = os.path.join('Images', file_name)
    
    # copying the image from the source to target file
    target = os.path.join(test_dir, label, file_name)

    copy(source, target)
```


```python

```
