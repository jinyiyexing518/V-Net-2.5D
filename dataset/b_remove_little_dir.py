import os


raw_train_path = "./data_raw_slice_tumour/train"
raw_label_path = "./data_raw_slice_tumour/label"
# raw_train_path = "./data_slice_tumour_modify_brightness/train"
# raw_label_path = "./data_slice_tumour_modify_brightness/label"

raw_dirs = os.listdir(raw_train_path)
raw_dirs.sort(key=lambda x: int(x))

for dir in raw_dirs:
    train_dir_path = os.path.join(raw_train_path, dir)
    label_dir_path = os.path.join(raw_label_path, dir)

    slice_num = len(os.listdir(train_dir_path))
    if slice_num < 20:

        trains = os.listdir(train_dir_path)
        trains.sort(key=lambda x: int(x.split('.')[0]))

        for name in trains:
            train_img_path = os.path.join(train_dir_path, name)
            label_img_path = os.path.join(label_dir_path, name)
            os.remove(train_img_path)
            os.remove(label_img_path)

        os.removedirs(train_dir_path)
        os.removedirs(label_dir_path)








