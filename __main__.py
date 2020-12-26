from __future__ import print_function

# import os
import torch
import argparse
import numpy as np

# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import datasets

# from tqdm import tqdm
# from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

# from datasets.vocabulary import Vocabulary

BATCH_SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument("--download",
                    action="store_true",
                    help="force download the dataset")

parser.add_argument("--verbose",
                    type=int,
                    help="set output verbosity",
                    default=0)

parser.add_argument("--train",
                    action="store_true",
                    help="use the train dataset")

if __name__ == "__main__":
    # ---------------------- Commander for the program -----------------------#

    args = parser.parse_args()

    df = datasets.VqaDataFolder(force=args.download,
                                verbose=args.verbose,
                                train=args.train)
    # dataset = fryday_ds.Daquar(daquar_processed_paths, transform)
    # trainloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    # )
    """
    image_size = (480, 640)
    transform = transforms.Compose(
        [
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    datafolder = fryday_ds.VqaDataFolder(
        force=args.download, verbose=args.verbose, train=args.train
    )
    dataset = fryday_ds.VQA(datafolder, transform)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    for batch_images, targets in trainloader:
        print(batch_images.shape)
    # for batch_idx, (v, q, a) in enumerate(trainloader):
    #     print(v[0])
    #     # print(v, q, a)
    #     break
    """

