import numpy as np
import time

import capnp
import msg_capnp
import msgpack
import msgpack_numpy as m

def array_to_cap(array, epoch=-1, batch=-1):
  array_cap = msg_capnp.TensorArray.new_message()
  array_cap.epoch = epoch
  array_cap.batch = batch
  
  inner_array_cap = array_cap.init('array', len(array))
  
  for i, (k, v) in enumerate(array):
    inner_array_cap[i].name = k
    inner_array_cap[i].data = msgpack.packb(v, default=m.encode)
    
  return array_cap

def cap_to_array(cap):
  array = []
  for tensor_cap in cap.array:
    array.append((tensor_cap.name, 
                  msgpack.unpackb(tensor_cap.data, object_hook=m.decode)))
  return array

def range_to_cap(start, end, name='worker', id=-1):
  return msg_capnp.WorkerInfo.new_message(start=start, end=end, 
                                          name=name, id=id)

def msg_to_cap(s):
  return msg_capnp.Msg.new_message(msg=s)
