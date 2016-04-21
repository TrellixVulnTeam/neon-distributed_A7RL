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
AllCNN style convnet on CIFAR10 data.

Reference:
    Striving for Simplicity: the All Convolutional Net `[Springenberg2015]`_
..  _[Springenberg2015]: http://arxiv.org/pdf/1412.6806.pdf
"""

from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import ModelDist
from neon.data import ArrayIterator, load_cifar10
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

class ModelCifar10AllCNN():

    def __init__(self):
        # parse the command line arguments
        parser = NeonArgparser(__doc__)
        parser.add_argument("--learning_rate", default=0.05, help="initial learning rate")
        parser.add_argument("--weight_decay", default=0.001, help="weight decay")
        parser.add_argument('--deconv', action='store_true',
                            help='save visualization data from deconvolution')
        self.args = parser.parse_args()

        # hyperparameters
        self.num_epochs = self.args.epochs

    def load_data(self):
        (X_train, y_train), (X_test, y_test), nclass = load_cifar10(path=self.args.data_dir,
                                                                    normalize=False,
                                                                    contrast_normalize=True,
                                                                    whiten=True)

        # really 10 classes, pad to nearest power of 2 to match conv output
        self.train_set = ArrayIterator(X_train, y_train, nclass=16, lshape=(3, 32, 32))
        self.valid_set = ArrayIterator(X_test, y_test, nclass=16, lshape=(3, 32, 32))

    def build_model(self):
        init_uni = Gaussian(scale=0.05)
        self.opt_gdm = GradientDescentMomentum(learning_rate=float(self.args.learning_rate), 
                                               momentum_coef=0.9,
                                               wdecay=float(self.args.weight_decay),
                                               schedule=Schedule(step_config=[200, 250, 300], 
                                                                 change=0.1))

        relu = Rectlin()
        conv = dict(init=init_uni, batch_norm=False, activation=relu)
        convp1 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1)
        convp1s2 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1, strides=2)

        layers = [Dropout(keep=.8),
                  Conv((3, 3, 96), **convp1),
                  Conv((3, 3, 96), **convp1),
                  Conv((3, 3, 96), **convp1s2),
                  Dropout(keep=.5),
                  Conv((3, 3, 192), **convp1),
                  Conv((3, 3, 192), **convp1),
                  Conv((3, 3, 192), **convp1s2),
                  Dropout(keep=.5),
                  Conv((3, 3, 192), **convp1),
                  Conv((1, 1, 192), **conv),
                  Conv((1, 1, 16), **conv),
                  Pooling(8, op="avg"),
                  Activation(Softmax())]

        self.cost = GeneralizedCost(costfunc=CrossEntropyMulti())

        self.model = ModelDist(layers=layers)

        """
        if args.model_file:
            import os
            assert os.path.exists(args.model_file), '%s not found' % args.model_file
            mlp.load_params(args.model_file)
        
        # configure callbacks
        callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)
        
        if args.deconv:
            callbacks.add_deconv_callback(train_set, valid_set)
        """
        
    def fit_worker(self):
        #self.model.fit_worker(self.train_set, optimizer=self.opt_gdm, num_epochs=self.num_epochs, 
        #                      cost=self.cost, callbacks=None)
        self.model.init2(self.train_set, self.cost, self.opt_gdm)
        self.model.event_init.set()
        #callbacks.on_train_begin(num_epochs)
        epoch_index = 0
        finished = False
        while epoch_index < self.num_epochs and not finished:
            self.model.total_cost[:] = 0

            for mb_idx, (x, t) in enumerate(self.train_set):
                # block until receive variables
                print 'wait on vars'
            
                self.model.event_recv.wait() # unblock by rpc runStep
                self.model.event_recv.clear()

                # var loaded
                self.model.compute_grads(x, t)
                # send out gradients
                self.model.event_send.set()
                
                # done
                print 'batch done'

            epoch_index += 1        
        

    def fit_ps(self):
        self.model.fit_ps(self.train_set, optimizer=self.opt_gdm, num_epochs=self.num_epochs, 
                          cost=self.cost, callbacks=None)

    def eval(self):
        print('Misclassification error = %.1f%%' % 
              (self.model.eval(self.valid_set, metric=Misclassification())*100))
