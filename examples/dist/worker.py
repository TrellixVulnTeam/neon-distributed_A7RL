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

import threading
#from model_mnist import ModelMnist
from cifar10_allcnn import ModelCifar10AllCNN

#======================= RPC server Class Def ===========================
class WorkerImpl(msg_capnp.Worker.Server):

  def __init__(self, model):
    self.iter_nr = 0
    self.MINIBATCH_SIZE = 100

    self.model = model
    self.model.build_model()
    
    self.event_recv = threading.Event()
    self.event_send = threading.Event()
    self.event_load_data = threading.Event()
    self.event_init = threading.Event()
    self.model.model.set_sync_event(self.event_recv, 
                                    self.event_send,
                                    self.event_init)

  def run_model(self):
    print 'wait for loading data'
    self.event_load_data.wait()
    self.model.fit_worker()

  # cap
  def loadData(self, info, **kwargs):
    print info.start, info.end
    self.model.load_data()
    #self.model.load_data(info.start, info.end)
    self.event_load_data.set()
    return msg_to_cap("load done")

  # cap
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
    self.event_init.wait()
    self.model.model.load_vars(var_array)

    # vars received, continue fit
    self.model.model.event_recv.set()
    
    self.iter_nr += 1
    
    # block here, wait for grads to be ready
    self.event_send.wait();
    self.event_send.clear()
    print "%f: Run done" % time.time()

    # extract grad_array here
    grad_array = self.model.model.get_grads()
    #print grad_array

    return array_to_cap(grad_array)

  def endRun(self, **kwargs):
    print 'run ends'
    return msg_to_cap("end done")


# build model
#model = ModelMnist()
model = ModelCifar10AllCNN()

# run rpc client
worker = WorkerImpl(model)
server = capnp.TwoPartyServer('*:60000', bootstrap=worker)

def worker_thread_fn(server):
  print 'Worker is ready'
  worker.run_model()

thread = threading.Thread(target=worker_thread_fn, args=[worker])
thread.start()

print 'Server is running'
server.run_forever()
thread.join()
