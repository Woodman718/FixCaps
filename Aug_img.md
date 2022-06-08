```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os
from shutil import copy, rmtree 
import tensorflow as tf
# import cv2
```
lesion_type_dict = {
    'nv': 'Melanocytic_nevi',
    'mel': 'melanoma',
    'bkl': 'Benign_keratosis-like_lesions',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma'
}lesion_danger = {
    'nv': 0, # 0 for benign
    'mel': 1, # 1 for malignant
    'bkl': 0, # 0 for benign
    'bcc': 1, # 1 for malignant
    'akiec': 1, # 1 for malignant
    'vasc': 0,
    'df': 0
}

```python
# targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# # To rename documents before action.
# # targetnames = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
# train_dir = "train50per/"
```


```python
def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)
```


```python
# source directory
cwd = os.getcwd()
data_root = os.path.abspath(os.path.join(cwd))
origin_data_path = os.path.join(data_root, "train501")
assert os.path.exists(origin_data_path), "path '{}' does not exist.".format(origin_data_path)
```


```python
data_class = [cla for cla in os.listdir(origin_data_path)
                if os.path.isdir(os.path.join(origin_data_path, cla))]
data_class
```




    ['vasc', 'nv', 'bkl', 'akiec', 'mel', 'df', 'bcc']




```python
# Augmentation directory
train_root = os.path.join(data_root,"train501")
# mk_file(train_root)
# for cla in data_class:
#     mk_file(os.path.join(train_root, cla))
# !ls {train_root}
```


```python
origin_data_path
```




    '/home/woodman/Jupyter/songbai/data/train501'




```python
train_root
```




    '/home/woodman/Jupyter/songbai/data/train501'




```python
# Augmenting images and storing them in temporary directories 
for img_class in data_class:

    #creating temporary directories
    # creating a base directory
    aug_dir = "aug_dir"   
    # creating a subdirectory inside the base directory for images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')

    mk_file(img_dir)
    
    cla_path = os.path.join(origin_data_path,img_class)
    img_list = os.listdir(cla_path)

    # Copy images from the class train dir to the img_dir 
    for index, image in enumerate(img_list):
        # path of source image in training directory
        image_path = os.path.join(cla_path,image)
        # creating a target directory to send images 
        tag_path = os.path.join(data_root,img_dir,image)
        # copying the image from the source to target file
        copy(image_path, tag_path)
        
    # Temporary augumented dataset directory.
    source_path = os.path.join(data_root,aug_dir)
    # Augmented images will be saved to training directory
    save_path = os.path.join(train_root,img_class)

    # Creating Image Data Generator to augment images
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(source_path,save_to_dir=save_path,save_format='jpg',save_prefix='trans_',target_size=(299, 299),batch_size=batch_size)

    # Generate the augmented images. Default:8000->51699
    aug_images = 8000 #29263
    
    num_files = len(img_list)
    num_batches = int(np.ceil((aug_images - num_files) / batch_size))

    # creating 8000 augmented images per class
    for i in range(0, num_batches):
        images, labels = next(aug_datagen)

    # delete temporary directory 
    rmtree(aug_dir)
```

    Found 132 images belonging to 1 classes.
    Found 6042 images belonging to 1 classes.
    Found 1033 images belonging to 1 classes.
    Found 304 images belonging to 1 classes.
    Found 1079 images belonging to 1 classes.
    Found 109 images belonging to 1 classes.
    Found 488 images belonging to 1 classes.



```python
# # copy origin_data_path(9187) to train_root().
# total_num = 0
# for cla in data_class:

#     cla_path = os.path.join(origin_data_path, cla)
#     images = os.listdir(cla_path)
#     num = len(images)
#     total_num += num
#     for index, image in enumerate(images):
#         image_path = os.path.join(cla_path, image)
#         img_name = image_path.split('/')[-1].split(".")[0]
#         savepath = os.path.join(train_root, cla,img_name + ".jpg")

#         img = Image.open(image_path)
#         img = img.resize((299, 299))#, resample=Image.LANCZOS)
#         img.save(savepath,quality=100)
#         # png
#         # cv2.imwrite(savepath,img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
#         # cv2.resize()
#         # jpg
#         # cv2.imwrite(savepath,img,[int(cv2.IMWRITE_JPEG_QUALITY),100])

#         print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
#     # break
#     print()

# print(f"processing {total_num} done!")
```


```python
# detect 
total_num = 0
for cla in data_class:
    cla_path = os.path.join(train_root, cla)
    images = os.listdir(cla_path)
    num = len(images)
    total_num += num
    for index, image in enumerate(images):
 
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    # break
    print()

print(f"processing {total_num} done!")
```

    [vasc] processing [7096/7096]
    [nv] processing [8042/8042]
    [bkl] processing [7931/7931]
    [akiec] processing [6992/6992]
    [mel] processing [7903/7903]
    [df] processing [5877/5877]
    [bcc] processing [7858/7858]
    processing 51699 done!



```python

```
