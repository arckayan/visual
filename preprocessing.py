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
import h5py
import torch

# Module under  this repository
import net
import log

from tqdm import tqdm

def process_vf(dataloader, options):
    if os.path.exists(options.vf_file):
        log._L(
            "Features file is already present, use `--verify-vf` to verify the features in the file"
        )

        # do not process is the file is already present
        return

    # Shape for the feature dataset (x, 2048, 14, 14)
    shape = (len(dataloader.dataset), options.output_feature,
             options.output_size, options.output_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log._L("preprocessing images using {}".format(log._P(device)))

    model = net.Resnet()
    model.to(device)
    model.eval()

    # process images using resent and store them in a h5 filep
    with h5py.File(options.vf_file, mode='w', libver='latest') as fd:
        # datasets for holding extracted features and corresponding ids
        features = fd.create_dataset('features', shape=shape, dtype='float16')
        image_ids = fd.create_dataset('ids',
                                      shape=(len(dataloader.dataset), ),
                                      dtype='int32')

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


def verify_vf(options):
    if os.path.exists(options.vf_file):
        log._D("Features file does not exists, try running `--process-vf`.")

        # do not verify as the feature file is missing
        return

    with h5py.File(options.vf_file, mode='r') as fd:
        for i in range(len(fd['ids'])):
            if fd['ids'][i] == 0:
                log._D("Error at index {} in feature file".format(i))
                break
