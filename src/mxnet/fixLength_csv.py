#coding=utf-8
import sys
import numpy as np
import mxnet as mx
import random


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_feature(n):
    ret = np.zeros(10)
    ret[n] = 1
    return ret

def gen_rand():
    num = random.randint(0, 9999)
    buf = str(num)
    while len(buf) < 4:
        buf = "0" + buf
    ret = np.array([])
    for i in range(80):
        c = int(buf[i / 20])
        ret = np.concatenate([ret, gen_feature(c)])
    return buf, ret

def get_label(buf):
    ret = np.zeros(4)
    for i in range(4):
        ret[i] = 1 + int(buf[i])
    return ret

class DataIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, init_states):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.count = count
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 10 * 80))] + init_states
        self.provide_label = [('label', (self.batch_size, 4))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_rand()
                data.append(img)
                label.append(get_label(num))
                
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']
            
            
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

