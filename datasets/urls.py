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
VQA_QA           = 'qa'
VQA_QA_TEST      = 'qa_test'
VQA_QA_TRAIN     = 'qa_train'

VQA_IM           = 'im'
VQA_IM_TEST      = 'im_test'
VQA_IM_TRAIN     = 'im_train'
VQA_QA_Q         = 'im_qa_train'
VQA_QA_ANOT      = 'im_qa_anot'


VQA_URLS = {
    VQA_QA_ANOT  : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip",
    VQA_QA_Q     : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip",
    VQA_IM       : "http://images.cocodataset.org/zips/train2014.zip",
}

