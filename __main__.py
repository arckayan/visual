from __future__ import print_function

# import os
import torch
import argparse
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import datasets as fryday_ds
# from tqdm import tqdm
# from PIL import Image
from torchvision import transforms
# from datasets.vocabulary import Vocabulary

BATCH_SIZE = 64

if __name__ == "__main__":
    # ---------------------- Commander for the program -----------------------#
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--download",
    #                     help="force download the dataset",
    #                     default=False)
    # parser.add_argument("--verbose",
    #                     help="set output verbosity",
    #                     default=0)

    # args = parser.parse_args()

    # daquar_processed_paths = fryday_ds.DaquarDataFolder(
    #     force=args.download, verbose=args.verbose
    # ).paths()
    
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    # )
    # print(daquar_processed_paths)
    # dataset = fryday_ds.Daquar(daquar_processed_paths, transform)
    # trainloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    # )
    k = fryday_ds.VqaDataFolder()

