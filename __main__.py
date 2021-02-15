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

import os
import argparse
import torch

import log
import net
import datasets
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # '--download' argument downloads the dataset in the provided or default path
    parser.add_argument("--download",
                        action="store_true",
                        help="download the dataset from the web")

    # '--path' argument specifies the dataset path
    parser.add_argument("--path",
                        default=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/vqa'),
                        help="path where the dataset is loaded")

    # '--split' argument is for specifying the dataset split to use, can be combined
    # preprocessing and downloading.
    parser.add_argument("--split",
                        default='valid',
                        help="specify the split for the dataset")

    # '--preprocess' argument genreates the image feature database from of vqa images
    # using a resnet model.
    parser.add_argument("--preprocess",
                        action="store_true",
                        help="preprocess the images and extract features using resnet")

    # '--verbose' argument is for detailed logging of all the processes, it takes
    # numbers(levels) upto 4, four being the level with the highest details.
    parser.add_argument("--verbose",
                        type=int,
                        default=0,
                        help="set output verbosity")


    # '--batch-size' is for specifying the dataloader's batch size
    parser.add_argument("--batch-size",
                        type=int,
                        default=64,
                        help="size of the batchs used for training")

    # '--image-size' specifies the size of the final input image (after cropping and
    # adding paddings)
    parser.add_argument("--image-size",
                        type=int,
                        default=448,
                        help="size of the input image to be fed into NN.")

    # '--central-fraction' is used when images are cropped, this tells the module how
    # much to scale
    parser.add_argument("--central-fraction",
                        type=int,
                        default=0.875,
                        help="Only take this much of the centre when scaling and cropping")

    # '--num-workers' specifies the number of workers for the pytorch dataloader
    parser.add_argument("--num-workers",
                        type=int,
                        default=8,
                        help="specifies the number of workers for the dataloader")

    args = parser.parse_args()
    args.output_size = args.image_size // 32
    args.output_feature = 2048
    # OUTPUT_SIZE = args.image_size // 32
    # OUTPUT_FEATURES = 2048 # same as number of features

    if args.download:
        # Download the dataset
        df = datasets.vqa.DataFolder(split=args.split, path=args.path)
        log._L("{} split downloaded at {}".format(log._S(args.split), log._P(args.path)))

    if args.preprocess:
        # Create a transformer for the the image - this will do the cropping and
        # add paddings to the image.
        transform = utils.create_transform(args.image_size, args.central_fraction)

        # Create an array of DataFolder of all the split
        dfs = [datasets.vqa.DataFolder(split=s, path=args.path) for s in ['train', 'test', 'valid']]

        # Array of Coco Image dataset for all splits
        ds = [datasets.Coco(df.paths()[0], transform=transform) for df in dfs]

        # We'll processes these together, therfore creating a composite Image
        # dataset
        composite_dataset = datasets.CocoComposite(*ds)

        # Create a dataloader for the dataset
        dataloader = torch.utils.data.DataLoader(composite_dataset,
                                                 shuffle=False,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

        # Extract Features from the composite_dataset images using resent or F-RCNN
        utils.preprocess_composite(dataloader, args)
