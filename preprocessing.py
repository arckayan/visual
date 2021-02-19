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
import re
import h5py
import json
import torch
import string
import itertools

# Module under  this repository
import net
import log

from collections import Counter
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



# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def process_punctuation(sen):
    if _punctuation.search(sen) is None:
        return sen
    s = _punctuation_with_a_space .sub('', sen)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()


def tokenize_questions(path):
    # TODO : use NLTK for processing and cleaning
    with open(path) as fd:
        questions_json = json.load(fd)
        questions = [q['question'] for q in questions_json['questions']]

        for question in questions:
            # prepare every question for processing - make it lower case
            # remove ? from the end
            question = process_punctuation(question.lower())
            # split the question into words and return the token
            yield question.split(' ')


def tokenize_answers(path):
    with open(path) as fd:
        answers_json = json.load(fd)
        answers = [[answer['answer'] for answer in ans_dict['answers']]
                   for ans_dict in answers_json['annotations']]

        for answer in answers:
            yield list(map(process_punctuation, answer))


def vocab_from_tokens(itr, top=None, min_value=0):
    tokens = itertools.chain.from_iterable(itr)
    counter = Counter(tokens)

    tokens = sorted(counter.keys() if top is None else
                    {t for t, c in counter.most_common(top)},
                    key=lambda x: (counter[x], x),
                    reverse=True)
    vocab = {t:i for i, t in enumerate(tokens, start=min_value)}
    return vocab


def process_tf(datafolder, options):
    questions_path = datafolder.paths()[1]
    questions = tokenize_questions(questions_path)

    if options.split != 'test':
        answers_path = datafolder.paths()[2]
        answers = tokenize_answers(answers_path)

    with open(options.tf_file, mode='w') as fd:
        json.dump(
            {
                "question": vocab_from_tokens(questions, min_value=1),
                "answer": vocab_from_tokens(answers, top=3000)
            }, fd)

