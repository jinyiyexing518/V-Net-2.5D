import os
import cv2 as cv
import numpy as np


raw_slice_train_path = "D:/pycharm_project/V_Net_tumour_processing_data/dataset/" \
                       "data_slice_tumour_modify_brightness/train"
raw_slice_label_path = "D:/pycharm_project/V_Net_tumour_processing_data/dataset/" \
                       "data_slice_tumour_modify_brightness/label"


train_clip_save_path = "./data_cv_clip_whole/train"
label_clip_save_path = "./data_cv_clip_whole/label"


dirs = os.listdir(raw_slice_train_path)
dirs.sort(key=lambda x: int(x))

j = 0
for dir in dirs:
    train_dir_path = os.path.join(raw_slice_train_path, dir)
    label_dir_path = os.path.join(raw_slice_label_path, dir)

    names = os.listdir(train_dir_path)
    names.sort(key=lambda x: int(x.split('.')[0]))

    i = 0
    for name in names:
        train_img_path = os.path.join(train_dir_path, name)
        label_img_path = os.path.join(label_dir_path, name)

        train_img = cv.imread(train_img_path, 0)
        label_img = cv.imread(label_img_path, 0)

        train_clip_img = train_img[0:400, 50:450]
        label_clip_img = label_img[0:400, 50:450]
        label_clip_img[label_clip_img == 255] = 255
        label_clip_img[label_clip_img != 255] = 0

        if label_clip_img.max() == 0:
            continue

        train_save_path = os.path.join(train_clip_save_path, dir)
        label_save_path = os.path.join(label_clip_save_path, dir)

        if not os.path.isdir(train_save_path):
            os.makedirs(train_save_path)
        if not os.path.isdir(label_save_path):
            os.makedirs(label_save_path)

        train_save_name = os.path.join(train_save_path, str(i) + ".png")
        label_save_name = os.path.join(label_save_path, str(i) + ".png")

        cv.imwrite(train_save_name, train_clip_img)
        cv.imwrite(label_save_name, label_clip_img)
        i += 1
    j += 1
    print("完成第{}个dir".format(j))





