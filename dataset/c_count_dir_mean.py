import os
import cv2 as cv
import numpy as np


train_raw_path = "./data_raw_slice_tumour/train"
label_raw_path = "./data_raw_slice_tumour/label"


def count_dir_mean_fun():
    dirs = os.listdir(train_raw_path)
    dirs.sort(key=lambda x: int(x))

    mean1 = []
    mean2 = 0.0
    mean3 = []

    for dir in dirs:
        train_dir_path = os.path.join(train_raw_path, dir)
        images = os.listdir(train_dir_path)
        images.sort(key=lambda x: int(x.split('.')[0]))

        image_num = len(images)
        mean = 0.0
        for name in images:
            image_path = os.path.join(train_dir_path, name)
            image = cv.imread(image_path, 0)
            image = image.astype("float32")
            # image /= 255

            black_pixel = image <= 5
            black_num = len(image[black_pixel])

            # mean += image.mean()
            mean += image.mean() * 512 * 512 / (512 * 512 - black_num)

        mean /= image_num
        mean1.append(mean)
    mean2 = sum(mean1) / len(mean1)
    mean3[:] = [x - mean2 for x in mean1]
    print(mean1)
    print(mean2)
    print(mean3)
    mean1 = np.array(mean1)
    mean2 = np.array(mean2)
    mean3 = np.array(mean3)
    # mean_save_path = "./mean_array"
    mean_save_path = "./mean_array_without_background"
    if not os.path.isdir(mean_save_path):
        os.makedirs(mean_save_path)
    np.save(mean_save_path + '/' + "mean1.npy", mean1)
    np.save(mean_save_path + '/' + "mean2.npy", mean2)
    np.save(mean_save_path + '/' + "mean3.npy", mean3)


if __name__ == "__main__":
    count_dir_mean_fun()




