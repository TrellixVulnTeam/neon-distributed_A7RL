#!/usr/bin/env python

import capnp
from cap.cap_helper import *

import threading, time
#from model_mnist import ModelMnist
from cifar10_allcnn import ModelCifar10AllCNN

import os
import db

os.system('service influxdb start')
db_server_host = 'choufong.ucsd.edu'

#======================= RPC server Class Def ===========================
class WorkerImpl(msg_capnp.Worker.Server):

  def __init__(self, model, db_client):
    self.iter_nr = 0
    self.MINIBATCH_SIZE = 128

    self.model = model
    self.model.build_model()
    
    self.event_recv = threading.Event()
    self.event_send = threading.Event()
    self.event_load_data = threading.Event()
    self.event_init = threading.Event()
    self.model.model.set_sync_event(self.event_recv, 
                                    self.event_send,
                                    self.event_init)
    self.db_client = db_client
    self.model.model.db_client = db_client
    
    self.name = 'unsigned'
    self.id = -1

  def run_model(self):
    print 'wait for loading data'
    self.event_load_data.wait()
    self.model.fit_worker()

  # cap
  def loadData(self, info, **kwargs):
    print '%s (%d) load data: %d - %d' % (info.name, info.id, 
                                          info.start, info.end)
    self.name = info.name
    self.id = info.id

    self.db_client.set_worker(info.name, info.id)

    print "%f: Load data start" % time.time()
    self.db_client.commit_cache('load', 's', -1, -1)

    self.model.load_data()
    #self.model.load_data(info.start, info.end)

    self.model.train_set.start = info.start
    self.model.train_set.ndata = info.end
    #print self.model.train_set.be.bsz

    self.event_load_data.set()
    print "%f: Load data end" % time.time()
    self.db_client.commit_cache('load', 'e', -1, -1)

    return msg_to_cap("load done")

  # cap
  def runStep(self, ins, **kwargs):
    epoch = ins.epoch
    batch = ins.batch
    
    print "%f: Run %d start" % (time.time(), self.iter_nr)
    self.db_client.commit_cache('vars', 'r', epoch, batch)
    
    #t_start = time.time()
    var_array = cap_to_array(ins)
    #t_end = time.time()
    #d_t = t_end - t_start

    # load variables
    self.event_init.wait()
    self.model.model.load_vars(var_array)
    print '%f: Load Vars' % time.time()
    self.db_client.commit_cache('vars', 'l', epoch, batch)
    
    # vars received, continue fit
    self.model.model.event_recv.set()
    self.iter_nr += 1
    
    # block here, wait for grads to be ready
    self.event_send.wait();
    self.event_send.clear()
    
    print "%f: Run done" % time.time()
    self.db_client.commit_cache('grads', 't', epoch, batch)

    # extract grad_array here
    grad_array = self.model.model.get_grads()
    return array_to_cap(grad_array)
    
  def endRun(self, **kwargs):
    print 'run ends'
    self.db_client.push_data()
    return msg_to_cap("end done")


# build model
#model = ModelMnist()
model = ModelCifar10AllCNN()

# influx db
db_client = db.DBClient(db_server_host, 8086, 'default', 'worker', 0)

# run rpc client
worker = WorkerImpl(model, db_client)
server = capnp.TwoPartyServer('*:60000', bootstrap=worker)

def worker_thread_fn(server):
  print 'Worker is ready'
  worker.run_model()

thread = threading.Thread(target=worker_thread_fn, args=[worker])
thread.start()

print 'Server is running'
server.run_forever()
thread.join()
