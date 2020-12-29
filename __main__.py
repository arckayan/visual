# Copyright (C) 2020 Manish Sahani <rec.manish.sahani@gmail.com>.
#
# This code is Licensed under the Apache License, Version 2.0 (the "License");
# A copy of a License can be obtained at:
#                 http://www.apache.org/licenses/LICENSE-2.0#
#
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --*= __main__.py =*----

import argparse
import h5py
import torch

import net
import datasets
import utils

from tqdm import tqdm

if __name__ == "__main__":

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

    args = parser.parse_args()
    OUTPUT_SIZE = args.image_size // 32
    OUTPUT_FEATURES = 2048 # same as number of features

    if args.download:
        ds = datasets.vqa.DataFolder(split=args.split)

    elif args.preprocess:
        train_df = datasets.vqa.DataFolder(split='train')
        valid_df = datasets.vqa.DataFolder(split='valid')

        transform = utils.create_transform(args.image_size, args.central_fraction)
        df = utils.coco_composite(transform, [train_df, valid_df])
        dl = torch.utils.data.DataLoader(df,
                                         shuffle=False,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         pin_memory=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = net.Resnet()
        model.to(device)
        model.eval()
        features_shape = (
            len(dl.dataset),
            OUTPUT_FEATURES,
            OUTPUT_SIZE,
            OUTPUT_SIZE
        )
        with h5py.File('./resnet14x14.h5', libver='latest') as fd:
            features = fd.create_dataset('features', shape=features_shape, dtype='float16')
            coco_ids = fd.create_dataset('ids', shape=(len(dl.dataset),), dtype='int32')
            i = j = 0
            for ids, imgs in tqdm(dl):
                imgs = imgs.cuda().clone().detach().requires_grad_(True)
                out = model(imgs)
                j = i + imgs.size(0)
                features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
                coco_ids[i:j] = ids.numpy().astype('int32')
                i = j
    else:
        df = datasets.vqa.DataFolder(split=args.split)
        tf = utils.create_transform(args.image_size, args.central_fraction)
        ds = datasets.vqa.RawVqa(df, tf)
        for x in ds:
            print(x[2].shape)
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

