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

import logging

from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator, load_mnist
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import ModelDist
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification
from neon.util.argparser import NeonArgparser

from neon.initializers.initializer import Uniform

import capnp
from cap.cap_helper import *

import threading

#======================= RPC server Class Def ===========================
class WorkerImpl(msg_capnp.Worker.Server):

  def __init__(self, model):
    self.iter_nr = 0
    self.MINIBATCH_SIZE = 100

    self.model = model
    self.event_recv = threading.Event()
    self.event_send = threading.Event()
    self.model.set_sync_event(self.event_recv, self.event_send)

  def run_model():
    self.model.fit()

  def loadData(self, info, **kwargs):
    self.model.load_data()
    #self.model.load_data(info.start, info.end)
    return msg_to_cap("done")

  def runStep(self, ins, **kwargs):
    print "%f: Run %d start" % (time.time(), self.iter_nr)

    t_start = time.time()
    # receive remote W, b here
    var_array = cap_to_array(ins)
    #print var_array
    t_end = time.time()
    d_t = t_end - t_start
    print "time cap_to_array: %f" % d_t    

    # load variables
    self.model.model.load_vars(var_array)

    # vars received, continue fit
    self.event_recv.set()

    print accuracy
    self.iter_nr += 1
    
    # block here, wait for grads to be ready
    self.event_send.wait();
    self.event_send.clear()
    print "%f: Run done" % time.time()

    # extract grad_array here
    grad_array = self.model.model.get_grads()
    #print grad_array

    return array_to_cap(grad_array)


#==================== build model ======================
class ModelMnist():
    def __init__():
        # parse the command line arguments
        parser = NeonArgparser(__doc__)

        self.args = parser.parse_args()

        self.logger = logging.getLogger()
        self.logger.setLevel(self.args.log_thresh)

    def load_data():
        # load up the mnist data set
        # split into train and tests sets
        (X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)

        # setup a training set iterator
        self.train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
        # setup a validation data set iterator
        self.valid_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))


    def build_model():
        # setup weight initialization function
        init_norm = Gaussian(loc=0.0, scale=0.01)

        # setup model layers
        layers = [Affine(nout=100, init=init_norm, bias=Uniform(), activation=Rectlin()),
                  Affine(nout=10, init=init_norm, bias=Uniform(), activation=Logistic(shortcut=True))]

        # setup cost function as CrossEntropy
        self.cost = GeneralizedCost(costfunc=CrossEntropyBinary())

        # setup optimizer
        self.optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9, stochastic_round=args.rounding)

        # initialize model object
        self.model = ModelDist(layers=layers)

        # configure callbacks
        self.callbacks = Callbacks(self.model, eval_set=self.valid_set, **self.args.callback_args)

    def fit():
        # run fit: run inside until finishing
        self.model.fit_worker(self.train_set, optimizer=self.optimizer, 
                              num_epochs=self.args.epochs, cost=self.cost, callbacks=self.callbacks)
        #print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification())*100))


# build model
model = ModelMnist()

# run rpc client
worker = WorkerImpl(model)
server = capnp.TwoPartyServer('*:60000', bootstrap=worker)
print "Worker is ready"

worker.run_model()
server.run_forever()
