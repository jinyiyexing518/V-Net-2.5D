import os
import cv2 as cv
import numpy as np


train_raw_path = "./data_raw_slice_tumour/train"
label_raw_path = "./data_raw_slice_tumour/label"


dirs = os.listdir(train_raw_path)
dirs.sort(key=lambda x: int(x))

train_save_path = "./data_slice_tumour_modify_brightness/train"
label_save_path = "./data_slice_tumour_modify_brightness/label"


# mean1 = np.load("./mean_array/mean1.npy")
# mean2 = np.load("./mean_array/mean2.npy")
mean1 = np.load("./mean_array_without_background/mean1.npy")
mean2 = np.load("./mean_array_without_background/mean2.npy")
for i in range(len(dirs)):
    dir_name = dirs[i]
    mean_sub = mean1[i]
    mean_add = mean2

    train_dir_path = os.path.join(train_raw_path, dir_name)
    images = os.listdir(train_dir_path)
    images.sort(key=lambda x: int(x.split('.')[0]))
    for name in images:
        image_path = os.path.join(train_dir_path, name)
        image = cv.imread(image_path, 0)
        image = image.astype("float32")
        # image /= 255 
        image -= mean_sub
        image += mean_add
        # image *= 255
        # image = image.astype("uint8")

        save_path = os.path.join(train_save_path, dir_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        cv.imwrite(save_path + '/' + name, image)
    print("完成第{}个dir".format(i))
