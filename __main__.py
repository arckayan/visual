# import os
import argparse
import torch

# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import net
import datasets
import utils
# from tqdm import tqdm
# from PIL import Image

# from datasets.vocabulary import Vocabulary

BATCH_SIZE = 64

parser = argparse.ArgumentParser()

parser.add_argument("--preprocess",
                    action="store_true",
                    help="preprocess the images and extract features using resnet")
parser.add_argument("--download",
                    action="store_true",
                    help="force download the dataset")
parser.add_argument("--verbose",
                    type=int,
                    default=0,
                    help="set output verbosity")
parser.add_argument("--split",
                    default='valid',
                    help="use the train dataset")
parser.add_argument("--batch-size",
                    type=int,
                    default=64,
                    help="size of the batchs used for training")
parser.add_argument("--image-size",
                    type=int,
                    default=448,
                    help="size of the input image to be fed into NN.")
parser.add_argument("--central-fraction",
                    type=int,
                    default=0.875,
                    help="Only take this much of the centre when scaling and cropping")
parser.add_argument("--num-workers",
                    type=int,
                    default=8,
                    help="specifies the number of workers for the dataloader")

if __name__ == "__main__":

    args = parser.parse_args()
    OUTPUT_SIZE = args.image_size // 32
    OUTPUT_FEATURES = 2048 # same as number of features

    if args.preprocess:
        train_df = datasets.vqa.DataFolder(split='train')
        valid_df = datasets.vqa.DataFolder(split='valid')

        transform = utils.create_transform(args.image_size, args.central_fraction)
        df = utils.coco_composite(transform, [train_df, valid_df])
        dl = torch.utils.data.DataLoader(df,
                                         shuffle=False,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         pin_memory=True)

        model = net.Resnet()
        model.eval()
        for batch in dl:
            from datetime import datetime
            print(datetime.now())
            #y = model(batch)
            print(y)
            print(datetime.now())

            break
    """
    preprocess_batch_size = 64
    image_size = 448  # scale shorter end of image to this size and centre crop
    output_size = image_size // 32  # size of the feature maps after processing through a network
    output_features = 2048  # number of feature maps thereof
    central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

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

