# Copyright (C) 2020 Manish Sahani.
#
# This code is Licensed under the Apache License, Version 2.0 (the "License");
# A copy of a License can be obtained at:
#                 http://www.apache.org/licenses/LICENSE-2.0#
#
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --*= urls.py =*----

###############################################################################
#
#   Daquar dataset urls
# 
###############################################################################
DAQUAR_QA           = 'qa'
DAQUAR_QA_TEST      = 'qa_test'
DAQUAR_QA_TRAIN     = 'qa_train'

DAQUAR_IM           = 'im'
DAQUAR_IM_TEST      = 'im_test'
DAQUAR_IM_TRAIN     = 'im_train'

DAQUAR_URLS = {
    DAQUAR_QA       : "https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.txt",
    DAQUAR_QA_TRAIN : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.train.txt",
    DAQUAR_QA_TEST  : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.test.txt",
    DAQUAR_IM_TRAIN : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/train.txt",
    DAQUAR_IM_TEST  : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/test.txt",
    DAQUAR_IM       : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar",
}

###############################################################################
#
#   VQA dataset urls
# 
###############################################################################

class VqaUrl:
    class Train:
        annotations = r"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
        questions = r"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
        images = r"http://images.cocodataset.org/zips/train2014.zip"
        # Names of the downloaded files
        file_an = annotations.split("/")[-1]
        file_qn = questions.split("/")[-1]
        file_im = images.split("/")[-1]
        # name of the extracted files
        V = 'train2014'
        Q = 'v2_OpenEnded_mscoco_train2014_questions.json'
        A = 'v2_mscoco_train2014_annotations.json'

    class Validation:
        annotations = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip'
        questions = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip'
        images = 'http://images.cocodataset.org/zips/val2014.zip'
        # Names of the downloaded files
        file_an = annotations.split("/")[-1]
        file_qn = questions.split("/")[-1]
        file_im = images.split("/")[-1]
        # name of the extracted files
        V = 'val2014'
        Q = 'v2_OpenEnded_mscoco_val2014_questions.json'
        A = 'v2_mscoco_val2014_annotations.json'

    class Test:
        questions = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip'
        images = 'http://images.cocodataset.org/zips/test2015.zip'
        # Names of the downloaded files
        file_qn = questions.split("/")[-1]
        file_im = images.split("/")[-1]
        # name of the extracted files
        V = 'test2015'
        Q = 'v2_OpenEnded_mscoco_test2015_questions.json'
