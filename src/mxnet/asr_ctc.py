# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
#coding=utf-8
import sys

import numpy as np
import mxnet as mx
import random
import logging
import codecs 
from lstm import lstm_unroll
from config_util import parse_args, parse_contexts, get_checkpoint_path
from fileTool import getLineNum
BATCH_SIZE = 3
SEQ_LENGTH = 5
LABEL_SIZE = 11

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

class FixLenCsvIter(mx.io.DataIter):
    def __init__(self, featFile, labelFile, batch_size, \
                 init_states, seq_len = 2000, frame_dim = 120, label_num = 2000, \
                 data_name='data', label_name='label', recordsNum2Read = -1):
        super(FixLenCsvIter, self).__init__()

        # pre-allocate with the largest bucket for better memory sharing
        self.featFile = featFile
        self.labelFile = labelFile

        self.featFilePtr = codecs.open(self.featFile, 'r', 'utf-8')
        self.labelFilePtr = codecs.open(self.labelFile, 'r', 'utf-8')

        self.batch_size = batch_size

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.seq_len = seq_len
        self.frame_dim = frame_dim
        self.label_num = label_num
        if recordsNum2Read > 0:
            self.recordsNum2Read = recordsNum2Read
        else:
            self.recordsNum2Read = getLineNum(featFile)
        assert(self.recordsNum2Read > 0)


        #self.provide_data = [('data', (batch_size, self.default_bucket_key))] + init_states
        #self.provide_label = [('label', (self.batch_size, self.default_bucket_key))]

        self.provide_data = [('data', (self.batch_size, self.seq_len * self.frame_dim))] + init_states
        self.provide_label = [('label', (self.batch_size, self.label_num))]
    

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        featLineNum = 0
        labelLineNum = 0
        recordsNumRead = 0
        batch_index = 0
        while recordsNumRead < self.recordsNum2Read:
            line4BatchRead = 0
            featMaxDimLen = -1
            labelMaxDimeLen = -1
            featBatchItems = []
            labelBatchItems = []

            # 1. read batch size items of feature
            for line in self.featFilePtr:
                featLineNum = featLineNum + 1
                line = line.strip()
                
                # 1. check empty line
                if len(line) <= 0:
                    print 'empty line on %d'%(featLineNum)
                    continue

                # 2. str to array
                splits = line.split(',')
                lenSplits = len(splits)
                assert(lenSplits % self.frame_dim == 0)
                assert(lenSplits < self.seq_len * self.frame_dim)

                item = [float(n) for n in splits]
                featBatchItems.append(item)
                
                # 3. judge uttrance read.
                line4BatchRead = line4BatchRead + 1
                if line4BatchRead >= self.batch_size:
                    break
            
            line4BatchRead = 0
            # 2. read batch size items of label
            for line in self.labelFilePtr:
                labelLineNum = labelLineNum + 1

                line = line.strip()
                
                # 1. check empty line
                if len(line) <= 0:
                    print 'empty line on %d'%(labelLineNum)
                    continue

                # 2. str to array
                splits = line.split(',')
                lenSplits = len(splits)
                assert(lenSplits < self.label_num)

                item = [float(n) for n in splits]
                labelBatchItems.append(item)

                # 3. judge uttrance label read.                
                line4BatchRead = line4BatchRead + 1

                if line4BatchRead >= self.batch_size:
                    break
            
            # 3. feature array to np.array
            data = np.zeros((self.batch_size, self.frame_dim * self.seq_len))
            
            for i in range(line4BatchRead):
                data[i][:len(featBatchItems[i])] = featBatchItems[i]

            # 4. label array to np.array
            label = np.zeros((self.batch_size, self.label_num))
            for i in range(line4BatchRead):
                label[i][:len(labelBatchItems[i])] = labelBatchItems[i]

            
            # 5. fill data for iterator
            data_all = [mx.nd.array(data)] + self.init_state_arrays    
            data_names = ['data'] + init_state_names

            # 6. fill label for iterator
            label_all = [mx.nd.array(label)]
            label_names = ['label']
            recordsNumRead = recordsNumRead + self.batch_size
            batch_index = batch_index + 1
            
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)

            yield data_batch


    def reset(self):
        self.featFilePtr.seek(0, 0)
        self.labelFilePtr.seek(0, 0)


def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret
        

def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH

    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = label[i]
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

if __name__ == '__main__':

    args = parse_args()
    config = args.config
    # port = config.getint('server', 'listenPort')
    #print 'port is %d'%(port)
    config.write(sys.stdout)
    
    # parameters for arch
    num_hidden = config.getint('arch', 'num_hidden')
#    num_hidden_proj = config.getint('arch', 'num_hidden_proj')
    num_lstm_layer = config.getint('arch', 'num_lstm_layer')
    
    # parameters for train
    recordsNum4Train = config.getint('train', 'recordsNum4Train')
    recordsNum4Test = config.getint('train', 'recordsNum4Test')
    batch_size = config.getint('train', 'batch_size')
    num_epoch = config.getint('train', 'num_epoch')
    learning_rate = config.getfloat('train', 'learning_rate')
    momentum = config.getfloat('train', 'momentum')
    prefix = config.get('train', 'prefix')

    # parameters for data
    train_feats = config.get('data', 'train_feats')
    train_labels = config.get('data', 'train_labels')
    dev_feats = config.get('data', 'dev_feats')
    dev_labels = config.get('data', 'dev_labels')
    seq_len = config.getint('data', 'seq_len')
    frame_dim = config.getint('data', 'frame_dim')
    label_num = config.getint('data', 'label_num')
    label_size = config.getint('data', 'label_size')
    trainRecords2Read = config.getint('train', 'recordsNum4Train')
    testRecords2Read = config.getint('train', 'recordsNum4Test')
    
    #label_num = seq_len
    
    BATCH_SIZE = batch_size
    SEQ_LENGTH = seq_len
    LABEL_SIZE = label_size
    contexts = parse_contexts(args)
    #contexts = [mx.context.gpu(i) for i in range(1)]
    assert(BATCH_SIZE % len(contexts) == 0)
    BATCH_SIZE = BATCH_SIZE / len(contexts)

    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')


    #contexts = [mx.context.gpu(0), mx.context.gpu(1)]
    contexts = parse_contexts(args)

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len,
                           num_hidden=num_hidden,
                           num_label = label_num,
                           label_size = LABEL_SIZE)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = FixLenCsvIter(train_feats, train_labels, batch_size, init_states, seq_len, frame_dim, label_num, recordsNum2Read = trainRecords2Read)
    data_val = FixLenCsvIter(dev_feats, dev_labels, batch_size, init_states, seq_len, frame_dim, label_num, recordsNum2Read = testRecords2Read)

    symbol = sym_gen(seq_len)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    print 'begin fit'

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy),
              batch_end_callback=mx.callback.Speedometer(batch_size, 5),epoch_end_callback = mx.callback.do_checkpoint(prefix))

    model.save(prefix)
