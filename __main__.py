# Copyright (C) 2020 Manish Sahani <rec.manish.sahani@gmail.com>.
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
import h5py
import torch

import log
import net
import utils
import datasets
import preprocessing

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # defaults
    _vqa = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/vqa')

    # '--download' (action) downloads the dataset in the path provided by the
    # user or in the default path (vqa_path)
    parser.add_argument("--download",
                        action="store_true",
                        help="download the dataset from the web.")

    # '--process-vf' (action) genreates the image feature database from of
    # vqa images using a resnet model.
    parser.add_argument("--process-vf",
                        action="store_true",
                        help="process and extract features from images.")

    # '--process-tf' (action) genreates the vocab from the question and ans.
    parser.add_argument("--process-tf",
                        action="store_true",
                        help="process the QA pairs and create a vocab.")

    # '--verify-vf' (action) verifies the features in feature file are processed
    # or not.
    parser.add_argument("--verify-vf",
                        action="store_true",
                        help="verify the features in the features file.")

    # '--path' (argument) specifies the dataset path where the dataset will be
    # downloaded and processed
    parser.add_argument("--path",
                        default=_vqa,
                        help="path where the dataset is loaded.")

    # '--split' (argument) is for specifying the dataset split, can be combined
    # downloading, traning and evaluating.
    parser.add_argument("--split",
                        default='valid',
                        help="specify the split for the dataset.")

    # '--verbose' (argument) is for detailed logging of all the processes, it
    # takes numbers(levels) upto 4, four being the level with the highest.
    parser.add_argument("--verbose",
                        type=int,
                        default=0,
                        help="set output verbosity for the program.")

    # '--batch-size' (argument) is for specifying the dataloader's batch size
    parser.add_argument("--batch-size",
                        type=int,
                        default=64,
                        help="size of the batchs used for training.")

    # '--image-size' (argument) specifies the size of the final input image
    # after cropping and adding paddings
    parser.add_argument("--image-size",
                        type=int,
                        default=448,
                        help="size of the input image to be fed into NN.")

    # '--central-fraction' (argument) is used when images are cropped, this
    # tells the module how much to scale
    parser.add_argument("--central-fraction",
                        type=int,
                        default=0.875,
                        help="take this much of the centre when cropping.")

    # '--num-workers' (argument) specifies the nu of workers for the dataloader
    parser.add_argument("--num-workers",
                        type=int,
                        default=8,
                        help="specifies the num of workers for the dataloader")

    # '--visual-features' (argument) is the processed feature database file of
    # coco images
    parser.add_argument("--vf-file",
                        default=os.path.join(_vqa, 'visual_features.h5'),
                        help="specifies the visual_features file")

    # '--vocab-most-common' (argument) is the top most common words choosen
    # while processing
    parser.add_argument("--vocab-most-common",
                        help="choose most common words while processing qa.")

    parser.add_argument("--tf-file",
                        default=os.path.join(_vqa, 'textual_vocab.json'),
                        help="specifies the textual vocab file")

    args = parser.parse_args()
    args.output_size = args.image_size // 32
    args.output_feature = 2048

    if args.download:
        # Download the dataset
        df = datasets.vqa.DataFolder(split=args.split, path=args.path)
        log._L("{} split downloaded at {}".format(log._S(args.split),
                                                  log._P(args.path)))

    if args.process_vf:
        # Create a transformer for the the image - this will do the cropping and
        # add paddings to the image.
        transform = utils.create_transform(args.image_size,
                                           args.central_fraction)

        # Create an array of DataFolder of all the split
        dfs = [
            datasets.vqa.DataFolder(split=s, path=args.path)
            for s in ['train', 'test', 'valid']
        ]

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
        preprocessing.process_vf(dataloader, args)

    if args.verify_vf:
        # verify the processed visual features in the vf_file
        preprocessing.verify_vf(args)

    if args.process_tf:
        # process textual features
        df = datasets.vqa.DataFolder(split=args.split, path=args.path)

        preprocessing.process_tf(df, args)
