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
import natsort

import torch
from PIL import Image
from torch.utils.data import Dataset

from log import _P, _L, _S
from datasets.vocabulary import Vocabulary
from datasets.downloader import wget
from datasets.urls import VqaUrl

class VqaDataFolder:
    """
    VqaDataFolder is the Handler for VQA dataset, includes the utilites to
    download images, question answer pairs and json files.
    """

    def __init__(self, path="./data/vqa", train=True, force=True, verbose=False):
        """Construct a brand new VQA Data Folder

        Args:
            path (str, optional): folders path. Defaults to "./data/vqa".
            train (bool, optional): for training set path. Defaults to True
            force (bool, optional): to force download. Defaults to False.
            verbose (bool, optional): detailed logs. Defaults to False.
        """
        self._path = os.path.abspath(path)
        self._train = train
        self._test = ~train
        self._force = force
        self._verbose = verbose
        self._downloader = wget
        self.urls = VqaUrl

        if self._force or not os.path.exists(self._path):
            self._download()

    def _download(self):
        self._VL("Downloading " + _P("VQA") + " in " + _S(self._path))

        if self._train:
            self._VL("Downloading Training split for the dataset")
            self._downloader(self.urls.Train.annotations, self._path)
            self._downloader(self.urls.Train.questions, self._path)
            self._downloader(self.urls.Train.images, self._path)

        if self._test:
            self._VL("Downloading Testing split for the dataset")
            self._downloader(self.urls.Test.questions, self._path)
            self._downloader(self.urls.Test.images, self._path)

        self._VL("Downloading Validation split for the dataset")
        self._downloader(self.urls.Validation.annotations, self._path)
        self._downloader(self.urls.Validation.questions, self._path)
        self._downloader(self.urls.Validation.images, self._path)


        """
        self._PROCESSED_QO_TRAIN = "OpenEnded_mscoco_train2014_questions.json"
        self._PROCESSED_QM_TRAIN = "MultipleChoice_mscoco_train2014_questions.json"
        self._PROCESSED_AN_TRAIN = "mscoco_train2014_annotations.json"
        self._PROCESSED_IM_TRAIN = "train_images"

        # Paths for files downloaded during processing (these paths are for
        # processing purpose only will not be returned as result)

        # zip file name containing all the images
        self._zip_im_fullname = self._urls[VQA_IM].split("/")[-1]  # .zip
        self._zip_im_name = self._zip_im_fullname.split(".")[0]
        self._zip_im_path = self._abspath(self._zip_im_fullname)

        # zip file for annotations
        self._zip_an_fullname = self._urls[VQA_QA_ANOT].split("/")[-1]
        self._zip_an_name = self._zip_an_fullname.split(".")[0]
        self._zip_an_path = self._abspath(self._zip_an_fullname)

        # zip file for questions
        self._zip_q_fullname = self._urls[VQA_QA_Q].split("/")[-1]
        self._zip_q_name = self._zip_q_fullname.split(".")[0]
        self._zip_q_path = self._abspath(self._zip_q_fullname)

        self._dir_im_extracted_train = self._abspath(self._zip_im_name)

        # outputs
        self._dir_im_train = self._abspath(self._PROCESSED_IM_TRAIN)
        self._json_qo_train = self._abspath(self._PROCESSED_QO_TRAIN)
        self._json_qm_train = self._abspath(self._PROCESSED_QM_TRAIN)
        self._json_an_train = self._abspath(self._PROCESSED_AN_TRAIN)

        # if self._force or (
        #     not os.path.exists(self._dir_im_test)
        #     and not os.path.exists(self._dir_im_train)
        # ):
        if self._force or (not os.path.exists(self._dir_im_train)):
            self._download()
            # self._extract()
            # self._resolve_dirs()

    # useable api for the class

    def paths(self):
        if self._train:
            return (
                self._dir_im_train,
                self._json_qm_train,
                self._json_qo_train,
                self._json_an_train,
            )
        else:
            return None
        #     return self._dir_im_test, self._json_test

    # Helper functions

    def _download(self):
        self._VL("Downloading " + _P("VQA") + " in " + _S(self._path))
        download_dataset(self._urls, self._path)

    def _extract(self):
        self._VL("Extracting the zips in " + _P(self._abspath()))

        # extract with output according to the verbosity
        zips = [self._zip_im_path, self._zip_an_path, self._zip_q_path]
        for z in zips:
            self._VL("Extracting - " + _P(z))
            os.system(
                "unzip {} -d {} {}".format(
                    z, self._path, ">/dev/null 2>&1" if self._verbose < 2 else " ",
                )
            )

    def _resolve_dirs(self):
        self._VL("Resolving directories, deleting old and creating new")

        # Rename the intermediate folder to test_images
        os.system("mv {} {}".format(self._dir_im_extracted_train, self._dir_im_train))
        """
    def _VL(self, log):
        # print the log according to the given verbosity
        if self._verbose:
            _L(log)

    def _abspath(self, path=""):
        # Return the absolute path after joining with the data folder path
        return os.path.join(self._path, path)


###############################################################################
#
#   VQA's handler for pytorch, with in-built vocabulary for processing the
#   questions and answers.
#
###############################################################################


class VQA(Dataset):
    """
    VQA Dataset

    Args:
        Dataset (Dataset): Pytorch's dataset
    """

    def __init__(self, datafolder, transform):
        """Constructor for the VQA

        Args:
            datafolder (DataFolder): path pointing to dataset folder
            transform : transformer for the images
        """
        super(VQA, self).__init__()
        im, qm, qo, an = datafolder.paths()

        self._im = im
        self._qm = qm
        self._qo = qo
        self._an = an

        self.transform = transform

        with open(self._qo, "r") as qo_f:
            qo_josn = json.load(qo_f)
            self._qo_tuples = qo_josn["questions"]
        with open(self._an, "r") as an_f:
            an_josn = json.load(an_f)
            self._an_tuples = an_josn["annotations"]

        self.questions = {}
        self.answers = {}
        self.total_imgs = natsort.natsorted(os.listdir(self._im))

        for qo in self._qo_tuples:
            self.questions[qo["image_id"]] = qo
        for an in self._an_tuples:
            self.answers[an["image_id"]] = an

        self._vocab_qo = Vocabulary("qo")
        self._vocab_an = Vocabulary("an")

        self._build_vocab()

    def _build_vocab(self):
        for qo in self._qo_tuples:
            self._vocab_qo.add_sentence(qo["question"])
        for an in self._an_tuples:
            for a in an["answers"]:
                self._vocab_an.add_sentence(a["answer"])

    def _encode_question(self, question):
        vec = torch.zeros(self._vocab_qo._longest_sentence).long()
        for i, token in enumerate(question.split(" ")):
            vec[i] = self._vocab_qo.to_index(token)

        return vec

    def _encode_answers(self, answers):
        vec = torch.zeros(self._vocab_an._num_words)
        for ans in answers:
            a = ans['answer'].split(' ')
            for _a in a:
                idx = self._vocab_an.to_index(_a)
                if idx is not None:
                    vec[idx] = 1

        return vec

    def __len__(self):
        """
        returns the length of the dataset
        """
        return len(self.total_imgs)

    def __getitem__(self, idx):
        """
        return the item from the dataset
        COCO_train2014_000000581921
        """

        v = self.transform(
            Image.open(os.path.join(self._im, self.total_imgs[idx])).convert("RGB")
        )

        image_id = int(self.total_imgs[idx].split("_")[-1].lstrip("0").split(".jpg")[0])
        if image_id not in self.questions.keys():
            return -1, -1, -1
            return (
                v,
                torch.zeros(self._vocab_qo._longest_sentence),
                self._encode_answers([]),
            )

        q = self._encode_question(self.questions[image_id]["question"])
        a = self._encode_answers(self.answers[image_id]['answers'])
        return v, q, a

