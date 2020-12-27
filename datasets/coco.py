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

# --*= dataset/coco.py =*--

import os
import torch.utils.data as data
from PIL import Image

class Coco(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.images = {}
        self.ids = []

        for file in os.listdir(self.path):
            if not file.endswith('.jpg'): continue

            id = int(file.split('_')[-1].split('.')[0])
            self.images[id] = file

        self.ids = sorted(list(self.images.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = Image.open(os.path.join(self.path, self.images[id]))

        if self.transform:
            image = self.transform(image)

        return image

class Composite(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, idx):
        for d in self.datasets:
            if idx < len(d): return d[idx]

            idx -= len(d)

        raise IndexError('Index too large for composite dataset')



















