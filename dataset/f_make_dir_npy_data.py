"""
本程序制作上传服务器的肝脏切片数据集
一张张切片，按照dir分布
"""
import os
import cv2 as cv
import numpy as np


# 处理过床位窗宽的train图像
train_png_path = "./data_cv_clip/train"
# label标签
label_png_path = "./data_cv_clip/label"
train_dir_npy_save_path = "./data_dir_npy/train"
label_dir_npy_save_path = "./data_dir_npy/label"
if not os.path.isdir(train_dir_npy_save_path):
    os.makedirs(train_dir_npy_save_path)
if not os.path.isdir(label_dir_npy_save_path):
    os.makedirs(label_dir_npy_save_path)

train_dirs = os.listdir(train_png_path)
label_dirs = os.listdir(label_png_path)
train_dirs.sort(key=lambda x: int(x))
label_dirs.sort(key=lambda x: int(x))

j = 0
for dir in train_dirs:
    train_dir_path = os.path.join(train_png_path, dir)
    label_dir_path = os.path.join(label_png_path, dir)

    dir_length = len(os.listdir(train_dir_path))
    train_dir_npy = np.ndarray((dir_length, 400, 400, 1), dtype=np.uint8)
    label_dir_npy = np.ndarray((dir_length, 400, 400, 1), dtype=np.uint8)

    train_imgs = os.listdir(train_dir_path)
    label_imgs = os.listdir(label_dir_path)
    train_imgs.sort(key=lambda x: int(x.split('.')[0]))
    label_imgs.sort(key=lambda x: int(x.split('.')[0]))

    i = 0
    for img in train_imgs:
        train_img_path = os.path.join(train_dir_path, img)
        label_img_path = os.path.join(label_dir_path, img)
        train_img = cv.imread(train_img_path, 0)
        label_img = cv.imread(label_img_path, 0)

        # cv.imshow("train", train_img)
        # cv.imshow("label", label_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        train_img = np.reshape(train_img, (400, 400, 1))
        label_img = np.reshape(label_img, (400, 400, 1))
        train_dir_npy[i] = train_img
        label_dir_npy[i] = label_img

        i += 1

    np.save(train_dir_npy_save_path + "/" + str(j) + ".npy", train_dir_npy)
    np.save(label_dir_npy_save_path + "/" + str(j) + ".npy", label_dir_npy)
    j += 1
    print("第{}个文件夹".format(j))















