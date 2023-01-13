import argparse
import os
import json
import random
import numpy as np
import torch
from torchvision import transforms
import transformers
from transformers import SegformerForSemanticSegmentation

from utils.datasets import SynapseDataset, RandomGenerator
from trainer import train, test


transformers.logging.set_verbosity_error()

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--test", type=bool, default=False, help="run test only")
parser.add_argument("--pretrained_model", type=str, default="nvidia/mit-b0", help="pretrained model name")
parser.add_argument("--dataset", type=str, default="Synapse", help="name of dataset")
parser.add_argument("--data_path", type=str, default="../data", help="root dir for data")
parser.add_argument("--list_path", type=str, default="../lists", help="list dir")
parser.add_argument("--model_path", type=str, default="../model/SegFormer", help="dir to save model")
parser.add_argument("--output_path", type=str, default="../output/SegFormer", help="dir to save log and figures")
parser.add_argument("--img_size", type=int, default=512, help="input patch size of network input")
parser.add_argument("--max_epochs", type=int, default=150, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=16, help="batch_size per gpu")
parser.add_argument("--accumulate_steps", type=int, default=1, help="accumulate steps")
parser.add_argument("--base_lr", type=float, default=0.00006, help="learning rate")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
args = parser.parse_args()

MODEL_PATH = os.path.join(args.model_path, args.dataset, args.pretrained_model.split("/")[-1])
OUTPUT_PATH = os.path.join(args.output_path, args.dataset, args.pretrained_model.split("/")[-1])
DATA_PATH = os.path.join(args.data_path, args.dataset)
LIST_PATH = os.path.join(args.list_path, f"lists_{args.dataset}")


if __name__ == "__main__":
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 加载标签
    id2label = json.load(open(os.path.join(DATA_PATH, "id2label.json")))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # 加载模型
    print(f"Loading pretrained model {args.pretrained_model}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained_model,
        id2label=id2label,
        label2id=label2id,
        num_channels=1,
        ignore_mismatched_sizes=True
    )

    # 若有 GPU 则使用 GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 训练
    if not args.test:
        train_ds = SynapseDataset(
            base_dir=DATA_PATH,
            list_dir=LIST_PATH,
            split="train",
            transform=transforms.Compose(
                [RandomGenerator(output_size=[args.img_size, args.img_size])]
            ),
        )
        print(f"Start Training")
        print(f"The best model will be saved in {MODEL_PATH}")
        print(f"The training log will be saved in {OUTPUT_PATH}")
        train(
            model=model,
            train_ds=train_ds,
            device=device,
            batch_size=args.batch_size,
            accumulate_steps=args.accumulate_steps,
            epochs=args.max_epochs,
            patience=6,
            lr=args.base_lr,
            model_path=MODEL_PATH,
            output_path=OUTPUT_PATH,
        )

    # 加载最好的模型
    print(f"Loading best model from {MODEL_PATH}")
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "best.pth")))
    model.eval()

    # 测试
    test_ds = SynapseDataset(
        base_dir=DATA_PATH,
        list_dir=LIST_PATH,
        split="test",
    )
    print(f"Start Testing")
    print(f"The test result will be saved in {OUTPUT_PATH}")
    test(
        model=model,
        test_ds=test_ds,
        device=device,
        batch_size=args.batch_size,
        output_path=OUTPUT_PATH,
    )