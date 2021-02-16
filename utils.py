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

# --*= utils.py =*--

import os
import h5py
import torch
import numpy as np

import net
import log

from tqdm import tqdm
from torchvision import transforms


def create_transform(image_size, central_fraction):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size / central_fraction)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )


def preprocess_composite(dataloader, options):
    # Shape for the feature dataset (x, 2048, 14, 14)
    shape = (
        len(dataloader.dataset),
        options.output_feature,
        options.output_size,
        options.output_size
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log._L("preprocessing images using {}".format(log._P(device)))
    model = net.Resnet()
    model.to(device)
    model.eval()

    # process images using resent and store them in a h5 filep
    with h5py.File(os.path.join(options.path, 'visual_features.h5'), mode='w', libver='latest') as fd:
        # datasets for holding extracted features and corresponding ids
        features = fd.create_dataset('features', shape=shape, dtype='float16')
        image_ids = fd.create_dataset('ids', shape=(len(dataloader.dataset),), dtype='int32')

        i = j = 0
        c = 18720
        # iterate over dataloader and process batch of images
        for ids, imgs in tqdm(dataloader):
            j = i + imgs.size(0)
            imgs.requires_grad_(True)
            imgs = imgs.to(device)
            out = model(imgs)

            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            image_ids[i:j] = ids.numpy().astype('int32')
            i = j


def verify_preprocessing(options):
    with h5py.File(os.path.join(options.path, 'visual_features.h5'), mode='r') as fd:
        print(fd['features'][0].shape)
        for i in fd['ids']:
            if i == 0:
                print("found")
