#!/usr/bin/env python

import capnp
from cap.cap_helper import *
#from model_mnist import ModelMnist
from cifar10_allcnn import ModelCifar10AllCNN

import math
import threading

class ReduceGrads:
    def __init__(self, total_worker):
        self.grad_array = []
        self.lock = threading.Lock()
        self.total_worker = total_worker
        self.done_worker = 0

    def reduce(self, grad_array):
        self.lock.acquire()
        if self.done_worker == 0:
            self.grad_array = grad_array
        else:
            i = 0
            for (k0, v0), (k1, v1) in zip(self.grad_array, grad_array):
                if k0 != k1:
                    print "Error: gradient term missmatch!"
                    exit(1)
                try:
                    # check with __iadd__
                    v0 += v1
                except np.linalg.LinAlgError:
                    print "Error: gradient addition"
                    exit(1)

                self.grad_array[i] = (k0, v0)
                i += 1

        self.done_worker += 1
        #print self.grad_array
        self.lock.release()

    def done(self):
        return self.done_worker == self.total_worker

    def result(self):
        if self.done_worker != self.total_worker:
            print "Error: reducer not done => done %d/total %d" % (reducer.done_worker, 
                                                                   reducer.total_worker)
            exit(1)
        else:
            self.done_worker = 0
            return self.grad_array

class WorkerStub:
    def __init__(self, reducer, data_size, batch_size, name="no", 
                 host="localhost", port=60000):
        self.name = name
        self.host = host
        self.port = port
        
        self.data_size = data_size
        self.batch_size = batch_size
        self.reducer = reducer
        
    def connect(self):
        self.client = capnp.TwoPartyClient(('%s:%d'%(self.host, self.port)))
        self.cap = self.client.bootstrap().cast_as(msg_capnp.Worker)

    def load_data(self, total, id):
        each_total = self.data_size / total
        self.start = id * each_total
        self.end   = self.start + each_total
        self.id = id
        
        return self.cap.loadData(info=range_to_cap((self.start, self.end)))
        
    # TODO: use call back 'then'
    def run_step_sync(self, var_cap):
        return self.cap.runStep(ins=var_cap).then(worker_callback)
    
def worker_callback(result):
    grad_array = cap_to_array(result.outs)
    print len(grad_array)
    reducer.reduce(grad_array)
    print "%f: %d Grads Recv" % (time.time(), 0)

DATA_SIZE = 300

worker_addr_list = [("localhost", 60000, "Myself")]

DATA_PER_WORKER = DATA_SIZE / len(worker_addr_list)
MINIBATCH_PER_WORKER = 128

NR_BATCH = int( math.ceil( float(DATA_PER_WORKER) / MINIBATCH_PER_WORKER) )

# build model
#model = ModelMnist()
model = ModelCifar10AllCNN()
model.load_data()
model.build_model()
model.model.set_nbatches(NR_BATCH)

# ================== create workers  ================
worker_list = []
worker_nr = len(worker_addr_list)

reducer = ReduceGrads(worker_nr)

for worker_addr in worker_addr_list:
    if len(worker_addr) < 2 or len(worker_addr) > 3:
        print "Wrong worker address: (host, port, name='NA')"
        exit(1)
    if len(worker_addr) == 3:
        name = worker_addr[2]
    else:
        name = worker_addr[3]
    worker_list.append( WorkerStub(reducer, DATA_SIZE, MINIBATCH_PER_WORKER, name,  
                                   worker_addr[0], worker_addr[1]) )

for worker in worker_list:
    worker.connect()

model.model.set_worker(worker_list, reducer)

# run fit
print "%f: Load data at workers" % time.time()
promises = []
for id, worker in enumerate(worker_list):
    promises.append(worker.load_data(worker_nr, id))

for p in promises:
    p.wait()
print "%f: Load data done" % time.time() 

model.fit_ps()
model.eval()
