```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from shutil import copy, rmtree 
import tensorflow as tf
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
targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# To rename documents before action.
# targetnames = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
train_dir = "train_dir/"
```


```python
def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)
```


```python
# Augmenting images and storing them in temporary directories 
for img_class in targetnames:

    #creating temporary directories
    # creating a base directory
    aug_dir = 'aug_dir'   
    # creating a subdirectory inside the base directory for images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    
    mk_file(aug_dir)
    mk_file(img_dir)

    img_list = os.listdir(train_dir + img_class)

    # Copy images from the class train dir to the img_dir 
    for file_name in img_list:

        # path of source image in training directory
        source = os.path.join(train_dir + img_class, file_name)

        # creating a target directory to send images 
        target = os.path.join(img_dir, file_name)

        # copying the image from the source to target file
        copy(source, target)

    # Temporary augumented dataset directory.
    source_path = aug_dir

    # Augmented images will be saved to training directory
    save_path = train_dir + img_class

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

    batch_size = 20

    aug_datagen = datagen.flow_from_directory(source_path,save_to_dir=save_path,save_format='png',target_size=(299, 299),batch_size=batch_size)

    # Generate the augmented images. Default:8000->51699
    aug_images = 8000 #29263

    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((aug_images - num_files) / batch_size))

    # creating 8000 augmented images per class
    for i in range(0, num_batches):
        images, labels = next(aug_datagen)

    # delete temporary directory 
    rmtree('aug_dir')
```

    Found 304 images belonging to 1 classes.
    Found 488 images belonging to 1 classes.
    Found 1033 images belonging to 1 classes.
    Found 109 images belonging to 1 classes.
    Found 1079 images belonging to 1 classes.
    Found 6042 images belonging to 1 classes.
    Found 132 images belonging to 1 classes.



```python

```
