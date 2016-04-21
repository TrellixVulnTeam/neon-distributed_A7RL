#!/usr/bin/env python

import capnp
from cap.cap_helper import *

import threading, time
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

    self.model.train_set.start = info.start
    self.model.train_set.ndata = info.end
    #print self.model.train_set.be.bsz

    self.event_load_data.set()
    return msg_to_cap("load done")

  # cap
  def runStep(self, ins, **kwargs):
    print "%f: Run %d start" % (time.time(), self.iter_nr)

    #t_start = time.time()
    var_array = cap_to_array(ins)
    #t_end = time.time()
    #d_t = t_end - t_start

    # load variables
    self.event_init.wait()
    self.model.model.load_vars(var_array)
    print '%f: Load Vars' % time.time()

    # vars received, continue fit
    self.model.model.event_recv.set()
    
    self.iter_nr += 1
    
    # block here, wait for grads to be ready
    self.event_send.wait();
    self.event_send.clear()
    print "%f: Run done" % time.time()

    # extract grad_array here
    grad_array = self.model.model.get_grads()
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
