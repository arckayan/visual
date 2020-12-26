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

# --*= daquar.py =*----

import os
import json
import torch
import natsort


from PIL import Image
from torch.utils.data import Dataset

# dataset modules developed as part of this resarch project
from log import _P, _L, _S
from datasets.vocabulary import Vocabulary

# from datasets.downloader import download_dataset
from datasets.urls import DAQUAR_URLS
from datasets.urls import (
    DAQUAR_IM,
    DAQUAR_QA_TRAIN,
    DAQUAR_IM_TRAIN,
    DAQUAR_IM_TEST,
    DAQUAR_QA_TEST,
)


###############################################################################
#
#   Daquar is a major public and opensource dataset for visaul question answers
#   it has around 1500 images of total size ~450MB, and around 16K training que
#   -stion answers pair.
#
###############################################################################


class DaquarDataFolder:
    """
    DaquarDataFolder is the Handler for DAQUAR dataset, includes the utilites
    to download images, question answer pairs and txt files.

    Also has the data processing processing utils, including spliting of
    images, translating question answer pairs in a json.
    """

    def __init__(self, path="./data/daquar", train=True, force=False, verbose=False):
        """Construct a brand new Dqauar Data Folder

        Args:
            path (str, optional): folders path. Defaults to "./data/daquar".
            train (bool, optional): for training set path. Defaults to True
            force (bool, optional): to force download. Defaults to False.
            verbose (bool, optional): detailed logs. Defaults to False.
        """
        self._path = os.path.abspath(path)
        self._train = train
        self._force = force
        self._verbose = verbose

        self._urls = DAQUAR_URLS

        # Final processed outputs image directories and qa json for train, and
        # test data
        self._PROCESSED_IM_TEST = "test_images"
        self._PROCESSED_IM_TRAIN = "train_images"
        self._PROCESSED_QA_TEST = "qa_test.json"
        self._PROCESSED_QA_TRAIN = "qa_train.json"

        # Paths for files downloaded during processing (these paths are for
        # processing purpose only will not be returned as result)

        # tar file name containing all the images
        self._tar_fullname = self._urls[DAQUAR_IM].split("/")[-1]  # .tar
        self._tar_name = self._tar_fullname.split(".")[0]  # nyu_depth_images
        self._tar_path = self._abspath(self._tar_fullname)

        # file that list of images in train and test dataset
        self._txt_im_test_name = self._urls[DAQUAR_IM_TEST].split("/")[-1]
        self._txt_im_train_name = self._urls[DAQUAR_IM_TRAIN].split("/")[-1]
        self._txt_im_test_path = self._abspath(self._txt_im_test_name)
        self._txt_im_train_path = self._abspath(self._txt_im_train_name)

        # txt file that holds all the question answers pairs
        self._txt_qa_test_name = self._urls[DAQUAR_QA_TEST].split("/")[-1]
        self._txt_qa_train_name = self._urls[DAQUAR_QA_TRAIN].split("/")[-1]
        self._txt_qa_test_path = self._abspath(self._txt_qa_test_name)
        self._txt_qa_train_path = self._abspath(self._txt_qa_train_name)

        # Json file hold all the vqa tuples
        self._json_test = self._abspath(self._PROCESSED_QA_TEST)  # output
        self._json_train = self._abspath(self._PROCESSED_QA_TRAIN)  # output

        # image directories paths
        self._dir_im_extracted = self._abspath(self._tar_name)
        self._dir_im_test = self._abspath(self._PROCESSED_IM_TEST)  # output
        self._dir_im_train = self._abspath(self._PROCESSED_IM_TRAIN)  # output

        # Downloadin, and processing vqa

        if self._force or (
            not os.path.exists(self._dir_im_test)
            and not os.path.exists(self._dir_im_train)
        ):
            self._download()
            self._extract_images()
            self._resolve_dirs()
            self._process_images()

            if self._train:
                self._process_questions(self._txt_qa_train_path, self._json_train)
            else:
                self._process_questions(self._txt_qa_test_path, self._json_test)

    # useable api for the class

    def paths(self):
        """Return the paths of the processed and useable dataset

        Returns:
            [string, string]: paths for image directory and json
        """
        if self._train:
            return self._dir_im_train, self._json_train
        else:
            return self._dir_im_test, self._json_test

    # Helper functions

    def _abspath(self, path):
        # Return the absolute path after joining with the data folder path
        return os.path.join(self._path, path)

    def _VL(self, log):
        # print the log according to the given verbosity
        if self._verbose:
            _L(log)

    def _download(self):
        """Download the dataset from the web, urls are predefined in the config
        """

        self._VL("Downloading " + _P("DAQUAR") + " in " + _S(self._path))

    #     download_dataset(self._urls, self._path)

    def _extract_images(self):
        """Extract the downloaded images
        """

        self._VL("Extracting the images in " + _P(self._dir_im_extracted))

        # extract with output according to the verbosity
        os.system(
            "tar xvfj {} -C {} {}".format(
                self._tar_path,
                self._path,
                ">/dev/null 2>&1" if not self._verbose else " ",
            )
        )

    def _resolve_dirs(self):
        """Resolve directories, delete old directories and create new ones
        """

        self._VL("Resolving directories, deleting old and creating new")

        # Delete the existing directories
        os.system("rm -rf {} {}".format(self._dir_im_test, self._dir_im_train))

        # Rename the intermediate folder to test_images
        os.system("mv {} {}".format(self._dir_im_extracted, self._dir_im_test))

        # make train directory
        os.makedirs(self._dir_im_train, exist_ok=True)

    def _process_images(self):
        """Process the downloaded dataset, split the images into test & train
           directories, and remove the intermediate directories.

           The dataset is split according to the list provided in the files in
           the dataset with name test.txt and train.txt
        """

        self._VL(
            "Seperating files from {} to {}".format(
                _P(self._dir_im_test), _S(self._dir_im_train)
            )
        )

        # move images which are in the train.txt list to the newly created test
        # directories
        with open(self._txt_im_train_path) as training_images_list:
            for image in [line.rstrip("\n") for line in training_images_list]:

                # mv  files from from_ to to_
                os.system(
                    "mv {} {}".format(
                        os.path.join(self._dir_im_test, image) + ".png",
                        os.path.join(self._dir_im_train, image) + ".png",
                    )
                )

    def _process_questions(self, from_, to_):
        """Process the downloaded questions, convert the txt files in json of
           tuples of image_id, question, answers.

        Args:
            from_ (string): txt file to be processed
            to_ (string): processed json file
        """

        self._VL("Processing the questions and answers, and creating a JSON")

        with open(from_) as txt_file:
            processed_questions = []
            for idx in txt_file:
                idx_n = next(txt_file)

                q = idx.rstrip("\n")
                a = idx_n.rstrip("\n")

                # process questions
                processed = {
                    "image_id": q.split("image")[-1].split(" ?")[0],
                    "question": q.rsplit(" ", 4)[0].replace(",", " ").replace("'", " "),
                    "answers": [ans.strip() for ans in a.split(",")],
                }

                processed_questions.append(processed)

            with open(to_, "w", encoding="utf-8") as f:
                json.dump(processed_questions, f, ensure_ascii=False, indent=4)


###############################################################################
#
#   Daquar's handler for pytorch, with in-built vocabulary for processing the
#   questions and answers.
#
###############################################################################


class Daquar(Dataset):
    """
    V
    """

    def __init__(self, datafolder, transform):
        """Constructor for the Daquar

        Args:
            datafolder (DataFolder): path pointing to image directory
            transform : transformer for the images
        """

        im, qa = datafolder.paths()

        # paths for image directory and qa json file
        self._im = im
        self._qa = qa

        self._vocab_q = Vocabulary("questions")
        self._vocab_a = Vocabulary("answer")

        self._questions = {}
        self._answers = {}

        self.transform = transform
        self.total_imgs = natsort.natsorted(os.listdir(self._im))

        self._build_vocab()
        self._process_qa()

    def _build_vocab(self):
        with open(self._qa) as json_f:
            json_pairs = json.load(json_f)
            for pair in json_pairs:
                self._vocab_q.add_sentence(pair["question"])

                for ans in pair["answers"]:
                    self._vocab_a.add_sentence(ans)

    def _process_qa(self):
        with open(self._qa) as f:
            json_pairs = json.load(f)
            for pair in json_pairs:
                q = pair["question"]
                a = pair["answers"]

                self._questions[pair["image_id"]] = self._encode_question(q)
                self._answers[pair["image_id"]] = self._encode_answers(a)

    def _encode_question(self, question):
        vec = torch.zeros(self._vocab_q._longest_sentence).long()
        for i, token in enumerate(question.split(" ")):
            vec[i] = self._vocab_q.to_index(token)

        return vec

    def _encode_answers(self, answers):
        vec = torch.zeros(self._vocab_a._num_words)
        for ans in answers:
            idx = self._vocab_a.to_index(ans)
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
        """
        v = self.transform(
            Image.open(os.path.join(self._im, self.total_imgs[idx])).convert("RGB")
        )
        image_id = self.total_imgs[idx].split("image")[-1].split(".png")[0]
        if image_id not in self._questions.keys():
            return (
                v,
                torch.zeros(self._vocab_q._longest_sentence),
                self._encode_answers([]),
            )

        q = self._questions[image_id]
        a = self._answers[image_id]

        # return v, q, a
        return v, q, a
