import os
from glob import glob
import h5py
import numpy as np
from tqdm import tqdm
import random

RAW_DATA_PATH = "./data/rawLung"
PROCESSED_DATA_PATH = "./data/Lung"
LISTS_PATH = "./lists/lists_Lung"


def split_dataset(data_path, lists_path):
    os.makedirs(lists_path, exist_ok=True)

    patient_list = glob(f"{data_path}/img_dir/*")
    patient_list = [path.split("/")[-1] for path in patient_list]
    with open(f"{lists_path}/all.lst", "w") as f:
        for patient in patient_list:
            f.write(f"{patient}\n")

    index = list(range(len(patient_list)))
    random.shuffle(index)
    train_index = index[: int(len(index) * 0.8)]
    test_index = index[int(len(index) * 0.8) :]
    train_list = [patient_list[i] for i in train_index]
    test_list = [patient_list[i] for i in test_index]
    return train_list, test_list


def preprocess_data(image_list, label_list, training_data, lists_path):
    if training_data:
        os.makedirs(f"{PROCESSED_DATA_PATH}/train_npz", exist_ok=True)
        list_file = open(f"{lists_path}/train.txt", "w")
    else:
        os.makedirs(f"{PROCESSED_DATA_PATH}/test_vol_h5", exist_ok=True)
        list_file = open(f"{lists_path}/test.txt", "w")

    for image, label in tqdm(
        zip(image_list, label_list), total=len(image_list)
    ):
        # 从路径字符串 */XXXXXXXX/_YYYYYY__ZZZZZZZZZZ 中解析出编号 ZZZZZZZZZZ
        number = image.split("_")[-1]

        # 计算切片数量
        n_slices = len(os.listdir(image))
        assert n_slices > 0

        if training_data:
            for i in range(n_slices):
                # 加载图像和标签
                image_data = np.load(f"{image}/slice{i}_img.npy").astype(np.float32)
                label_data = np.load(f"{label}/slice{i}_mask.npy").astype(np.int8)
                # 保存为 npz 格式
                save_path = f"{PROCESSED_DATA_PATH}/train_npz/case{number}_slice{i:03d}.npz"
                np.savez_compressed(
                    save_path,
                    label=label_data,
                    image=image_data,
                )
                # 写入列表文件
                list_file.write(f"case{number}_slice{i:03d}\n")
        else:
            image_volume = np.zeros((n_slices, 224, 224), dtype=np.float32)
            label_volume = np.zeros((n_slices, 224, 224), dtype=np.int8)
            for i in range(n_slices):
                # 加载图像和标签
                image_data = np.load(f"{image}/slice{i}_img.npy").astype(np.float32)
                label_data = np.load(f"{label}/slice{i}_mask.npy").astype(np.int8)
                # 拼接
                image_volume[i] = image_data
                label_volume[i] = label_data
            # 保存为 h5 格式
            save_path = f"{PROCESSED_DATA_PATH}/test_vol_h5/case{number}.npy.h5"
            f = h5py.File(save_path, "w")
            f["image"] = image_volume.astype(np.float32)
            f["label"] = label_volume.astype(np.int8)
            f.close()
            # write to list file
            list_file.write(f"case{number}\n")


if __name__ == "__main__":
    # 把数据集随即切分为训练集和测试集
    train_list, test_list = split_dataset(RAW_DATA_PATH, LISTS_PATH)

    # 得到图像和标签文件路径的列表
    train_img_list = [
        img
        for patient in train_list
        for img in glob(f"{RAW_DATA_PATH}/img_dir/{patient}/*")
    ]
    train_label_list = [img.replace("img_dir", "ann_dir") for img in train_img_list]
    test_img_list = [
        img
        for patient in test_list
        for img in glob(f"{RAW_DATA_PATH}/img_dir/{patient}/*")
    ]
    test_label_list = [img.replace("img_dir", "ann_dir") for img in test_img_list]

    print("Preprocessing training data...")
    preprocess_data(train_img_list, train_label_list, training_data=True, lists_path=LISTS_PATH)
    print("Preprocessing testing data...")
    preprocess_data(test_img_list, test_label_list, training_data=False, lists_path=LISTS_PATH)
    print("Done!")
