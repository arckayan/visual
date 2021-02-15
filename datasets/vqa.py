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

# --*= vqa.py =*----

import os
import json

import torch
from PIL import Image
from torch.utils.data import Dataset

from log import _P, _L, _S
from datasets.vocabulary import Vocabulary
from datasets.downloader import wget, unzip
from datasets.urls import VqaUrl

class DataFolder:
    """
    VqaDataFolder is the Handler for VQA dataset, includes the utilites to
    download images, question answer pairs and json files.
    """

    def __init__(self,
                 path,
                 split='valid',
                 verbose=False):
        """Construct a brand new VQA Data Folder

        Args:
            path (str, optional): folders path. Defaults to "./data/vqa".
            split (str, optional): for training set path. Defaults to valid
            verbose (bool, optional): detailed logs. Defaults to False.
        """
        self.path = os.path.abspath(path)
        self.split = split
        self.urls = VqaUrl
        self._verbose = verbose

        if False in [os.path.exists(p) for p in self.paths()]:
            self._download()
            self._extract()

    def _download(self):
        self._VL("Downloading " + _P("VQA") + " in " + _S(self.path))

        if self.split == 'train':
            self._VL("Downloading Training split for the dataset")
            wget(self.urls.Train.annotations, self.path)
            wget(self.urls.Train.questions, self.path)
            wget(self.urls.Train.images, self.path)
        elif self.split == 'test':
            self._VL("Downloading Testing split for the dataset")
            wget(self.urls.Test.questions, self.path)
            wget(self.urls.Test.images, self.path)
        else:
            self._VL("Downloading Validation split for the dataset")
            wget(self.urls.Validation.annotations, self.path)
            wget(self.urls.Validation.questions, self.path)
            wget(self.urls.Validation.images, self.path)

    def _extract(self):
        self._VL("Extracting the zips in " + _P(self.path))

        if self.split == 'train':
            self._VL("Extracting Training split for the dataset")
            unzip(os.path.join(self.path, self.urls.Train.file_an),self.path)
            unzip(os.path.join(self.path, self.urls.Train.file_qn),self.path)
            unzip(os.path.join(self.path, self.urls.Train.file_im),self.path)
        elif self.split == 'test':
            self._VL("Extracting Testing split for the dataset")
            unzip(os.path.join(self.path, self.urls.Test.file_qn), self.path)
            unzip(os.path.join(self.path, self.urls.Test.file_im),self.path)
        else:
            self._VL("Extracting Validation split for the dataset")
            unzip(os.path.join(self.path, self.urls.Validation.file_an),self.path)
            unzip(os.path.join(self.path, self.urls.Validation.file_qn),self.path)
            unzip(os.path.join(self.path, self.urls.Validation.file_im),self.path)

    def paths(self):
        if self.split == 'train':
            return (
                os.path.join(self.path, self.urls.Train.V),
                os.path.join(self.path, self.urls.Train.Q),
                os.path.join(self.path, self.urls.Train.A)
            )
        elif self.split == 'test':
            return (
                os.path.join(self.path, self.urls.Test.V),
                os.path.join(self.path, self.urls.Test.Q)
            )
        else:
            return (
                os.path.join(self.path, self.urls.Validation.V),
                os.path.join(self.path, self.urls.Validation.Q),
                os.path.join(self.path, self.urls.Validation.A)
            )

    def _VL(self, log):
        # print the log according to the given verbosity
        if self._verbose:
            _L(log)


###############################################################################
#
#   VQA's handler for pytorch, with in-built vocabulary for processing the
#   questions and answers.
#
###############################################################################


class RawVqa(Dataset):
    """
    RawVqa Dataset has the raw images instead of image features extracted pre-
    viosuly in preprocessing.

    Args:
        Dataset (Dataset): Pytorch's dataset
    """

    def __init__(self, datafolder, transform):
        """Constructor for the RawVqa

        Args:
            datafolder (DataFolder): path pointing to dataset folder
            transform : transformer for the images
        """
        super(RawVqa, self).__init__()
        self.V, self.Q, self.A = datafolder.paths()
        self.transform = transform
        self.questions = {}
        self.questions_idx = []
        self.answers = {}
        self.images = {}
        self.vocab_q = Vocabulary("Q")
        self.vocab_a = Vocabulary("A")

        self.process_v()
        self.process_q()
        self.process_a()

    def process_v(self):
        for file in os.listdir(self.V):
            if not file.endswith('.jpg'):
                continue
            id = int(file.split('_')[-1].split('.')[0])
            self.images[id] = file

    def process_q(self):
        with open(self.Q, 'r') as f:
            question = json.load(f)['questions']
            for q in question:
                self.vocab_q.add_sentence(q['question'])
                self.questions[q['question_id']] = q

        self.questions_idx = list(self.questions.keys())

    def process_a(self):
        with open(self.A, 'r') as f:
            annotations = json.load(f)['annotations']
            for a in annotations:
                self.answers[a['question_id']] = a
                for answer in a['answers']:
                    self.vocab_a.add_word(answer['answer'])

    def _encode_question(self, question):
        vec = torch.zeros(self.vocab_q._longest_sentence).long()
        for i, token in enumerate(question.split(" ")):
            vec[i] = self.vocab_q.to_index(token)

        return vec

    def _encode_answers(self, answers):
        vec = torch.zeros(self.vocab_a._num_words)
        for answer in answers:
            index = self.vocab_a.to_index(answer)
            if index is not None:
                vec[index] += 1
        return vec

    def __len__(self):
        """
        returns the length of the dataset
        """
        return len(self.questions_idx)

    def __getitem__(self, idx):
        """
        return the item from the dataset
        """
        id = self.questions_idx[idx]
        question = self.questions[id]
        path = os.path.join(self.V, self.images[int(question['image_id'])])

        V = self.transform(Image.open(path).convert('RGB'))
        Q = self._encode_question(question['question'])
        A = self._encode_answers([a['answer'] for a in self.answers[id]['answers']])

        return V, Q, A
