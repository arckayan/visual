from __future__ import print_function

# import os
import torch
import argparse
import numpy as np

# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import net
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
    preprocess_batch_size = 64
    image_size = 448  # scale shorter end of image to this size and centre crop
    output_size = image_size // 32  # size of the feature maps after processing through a network
    output_features = 2048  # number of feature maps thereof
    central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size / central_fraction)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.Vqa(df, transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE)
    model = net.vqa.Vqa()
    if torch.cuda.is_available():
        print("Using GPU")
        model.to('cuda')
    for batch in loader:
        y = model(batch)
        print(y)
        break
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

