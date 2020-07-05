"""
本程序制作test测试数据 
"""
import os
import cv2 as cv
import numpy as np
# from vnet_25D.vnet11_1.test_model.vnet_test import dir_num

from test_model.vnet3_test import dir_num
# dir_num = 129
# 制作哪一个dir的数据
# dir_num -= 121
image_size = 400
depth = 3


train_png_path = "./data_cv_clip/test/train"

label_png_path = "./data_cv_clip/test/label"
train_64_input_path = "./vnet_" + str(depth) + "_1_test_npy/train"
label_64_input_path = "./vnet_" + str(depth) + "_1_test_npy/label"
if not os.path.isdir(train_64_input_path):
    os.makedirs(train_64_input_path)
if not os.path.isdir(label_64_input_path):
    os.makedirs(label_64_input_path)


train_dirs = os.listdir(train_png_path)
label_dirs = os.listdir(label_png_path)
train_dirs.sort(key=lambda x: int(x))
label_dirs.sort(key=lambda x: int(x))
# 选择需要制作数据的dir
# dir_name = label_dirs[dir_num]
dir_name = str(dir_num)
# train和label的dir地址
train_dir_path = os.path.join(train_png_path, dir_name)
label_dir_path = os.path.join(label_png_path, dir_name)

train_pngs = os.listdir(train_dir_path)
train_pngs.sort(key=lambda x: int(x.split(".")[0]))
label_pngs = os.listdir(label_dir_path)
label_pngs.sort(key=lambda x: int(x.split(".")[0]))

count_num = len(train_pngs) - depth + 1
train_npys = np.ndarray((count_num, depth, image_size, image_size, 1), dtype=np.uint8)
label_npys = np.ndarray((count_num, image_size, image_size, 1), dtype=np.uint8)

for i in range(len(train_pngs)):
    train_npy = np.ndarray((depth, image_size, image_size, 1), dtype=np.uint8)
    label_npy = np.ndarray((image_size, image_size, 1), dtype=np.uint8)
    if (i + depth-1) < len(train_pngs):
        label_img_path = os.path.join(label_dir_path, label_pngs[i+depth//2])
        label_img = cv.imread(label_img_path, 0)
        label_img = np.reshape(label_img, (image_size, image_size, 1))
        label_npy = label_img
        for j in range(depth):
            index = i + j
            train_img_path = os.path.join(train_dir_path, train_pngs[index])
            train_img = cv.imread(train_img_path, 0)
            train_img = np.reshape(train_img, (image_size, image_size, 1))
            train_npy[j] = train_img
        train_npys[i] = train_npy
        label_npys[i] = label_npy

train_input_path = os.path.join(train_64_input_path, str(dir_num))
label_input_path = os.path.join(label_64_input_path, str(dir_num))
if not os.path.isdir(train_input_path):
    os.makedirs(train_input_path)
if not os.path.isdir(label_input_path):
    os.makedirs(label_input_path)
np.save(train_input_path + "/" + str(dir_num) + ".npy", train_npys)
np.save(label_input_path + "/" + str(dir_num) + ".npy", label_npys)













