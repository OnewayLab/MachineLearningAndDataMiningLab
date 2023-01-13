import os
import argparse
from glob import glob
import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


RAW_DATA_PATH = "./data/rawSynapse"
PROCESSED_DATA_PATH = "./data/Synapse"

LABEL_REMAPPING = {1: 7, 2: 4, 3: 3, 4: 2, 5: 0, 6: 5, 7: 8, 8: 1, 9: 0, 10: 0, 11: 6, 12: 0, 13: 0}

def preprocess_data(image_files, label_files, training_data):
    if training_data:
        os.makedirs(f"{PROCESSED_DATA_PATH}/train_npz", exist_ok=True)
    else:
        os.makedirs(f"{PROCESSED_DATA_PATH}/test_vol_h5", exist_ok=True)

    MIN, MAX = -125, 275  # 将图像裁剪到 [-125, 275] 范围内

    for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
        # 从路径字符串 */DETXXXX01.nii.gz 中解析出编号 XXXX
        number = image_file.split("/")[-1][3:7]

        # 加载图像和标签
        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata().astype(np.int8)

        # 将图像裁剪到 [MIN, MAX] 并标准化
        image_data = np.clip(image_data, MIN, MAX)
        image_data = (image_data - MIN) / (MAX - MIN)

        # 重映射标签
        new_label_data = np.zeros(label_data.shape, dtype=np.int8)
        for k, v in LABEL_REMAPPING.items():
            new_label_data[label_data == k] = v
        label_data = new_label_data

        H, W, D = image_data.shape
        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))

        if training_data:
            # 对于训练集，从 3D 图像中提取 2D 切片并保存为 npz 格式
            for dep in range(D):
                save_path = (f"{PROCESSED_DATA_PATH}/train_npz/case{number}_slice{dep:03d}.npz")
                np.savez_compressed(
                    save_path,
                    label=label_data[dep, :, :].astype(np.int8),
                    image=image_data[dep, :, :].astype(np.float32)
                )
        else:
            # 对于测试集，将图像和标签保存为 h5 格式
            save_path = f"{PROCESSED_DATA_PATH}/test_vol_h5/case{number}.npy.h5"
            f = h5py.File(save_path, "w")
            f["image"] = image_data.astype(np.float32)
            f["label"] = label_data.astype(np.int8)
            f.close()


if __name__ == "__main__":
    # 获取图像和标签文件的列表
    image_files = sorted(glob(f"{RAW_DATA_PATH}/img/*.nii.gz"))
    label_files = sorted(glob(f"{RAW_DATA_PATH}/label/*.nii.gz"))

    print("Preprocessing training data...")
    preprocess_data(image_files, label_files, training_data=True)
    print("Preprocessing testing data...")
    preprocess_data(image_files, label_files, training_data=False)
    print("Done!")
