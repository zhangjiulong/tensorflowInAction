#coding=utf-8

import sys
import numpy as np
import mxnet as mx
from lstm import lstm_unroll
import logging

from bucket_csv import BucketSentenceIter
from config_util import parse_args, parse_contexts, get_checkpoint_path

BATCH_SIZE = 3

def Accuracy(label, pred):

    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = label[i]
        p = []
        for k in range(len(label)):
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
    num_hidden_proj = config.getint('arch', 'num_hidden_proj')
    num_lstm_layer = config.getint('arch', 'num_lstm_layer')
    
    # parameters for train
    batch_size = config.getint('train', 'batch_size')
    num_epoch = config.getint('train', 'num_epoch')
    learning_rate = config.getfloat('train', 'learning_rate')
    momentum = config.getfloat('train', 'momentum')

    # parameters for data
    train_feats = config.get('data', 'train_feats')
    train_labels = config.get('data', 'train_labels')
    dev_feats = config.get('data', 'dev_feats')
    dev_labels = config.get('data', 'dev_labels')
    
    #contexts = parse_contexts(args)
    contexts = [mx.context.gpu(i) for i in range(1)]

    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len,
                           num_hidden=num_hidden,
                           num_label = label_dim)
    

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    
    '''
    data_train = mx.io.CSVIter(data_csv=train_feats, data_shape=(2, feat_dim),
                               label_csv=train_labels, label_shape = (3, 1),
                               batch_size=batch_size)
    data_dev = mx.io.CSVIter(data_csv = dev_feats, data_shape=(2, feat_dim),
                             label_csv=dev_labels, label_shape = (3, 1),
                             batch_size=batch_size)
    '''

    data_train = BucketSentenceIter(train_feats, train_labels, batch_size, init_states)
    data_dev = BucketSentenceIter(dev_feats, dev_labels, batch_size, init_states)
    
    symbol = sym_gen

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    
    print 'begin fit'

    model.fit(X=data_train, eval_data=data_dev,
              eval_metric = mx.metric.np(Accuracy),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

    #model.save("asr001")
    
