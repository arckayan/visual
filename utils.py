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

from torchvision import transforms

from datasets.coco import Coco, Composite

def create_transform(image_size, central_fraction):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size / central_fraction)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )


def coco_composite(transform, datafolders):
    datasets = [Coco(df.paths()[0], transform=transform) for df in datafolders]

    return Composite(*datasets)


def preprocess(transform, datafolders):
     print(datafolders)

