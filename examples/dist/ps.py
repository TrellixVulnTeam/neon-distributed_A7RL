#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Example that trains a small multi-layer perceptron with fully connected layers on MNIST.

This example has some command line arguments that enable different neon features.

Examples:

    python mnist_mlp.py -b gpu -e 10
        Run the example for 10 epochs of mnist data using the nervana gpu backend

    python mnist_mlp.py --eval_freq 1
        After each training epoch the validation/test data set will be processed through the model
        and the cost will be displayed.

    python mnist_mlp.py --serialize 1 -s checkpoint.pkl
        After every iteration of training the model will be dumped to a pickle file named
        "checkpoint.pkl".  Changing the serialize parameter changes the frequency at which the
        model is saved.

    python mnist_mlp.py --model_file checkpoint.pkl
        Before starting to train the model, the model state is set to the values stored in the
        checkpoint file named checkpoint.pkl.
"""

import capnp
from cap.cap_helper import *
#from model_mnist import ModelMnist
from cifar10_allcnn import ModelCifar10AllCNN

# build model
#model = ModelMnist()
model = ModelCifar10AllCNN()
model.load_data()
model.build_model()

# run rpc client
client = capnp.TwoPartyClient('localhost:60000')
cap = client.bootstrap().cast_as(msg_capnp.Worker)
model.model.set_cap(cap)

# run fit
promise = cap.loadData(info=range_to_cap((1, 10)))
print promise.wait()
model.fit_ps()
model.eval()
