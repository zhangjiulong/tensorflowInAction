# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import numpy as np
import mxnet as mx
import codecs 


# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, featFile, labelFile, batch_size,
                 init_states, data_name='data', label_name='label'):
        super(BucketSentenceIter, self).__init__()

        # pre-allocate with the largest bucket for better memory sharing
        self.featFile = featFile
        self.labelFile = labelFile
        self.default_bucket_key = 999
        self.featFilePtr = codecs.open(self.featFile, 'r', 'utf-8')
        self.labelFilePtr = codecs.open(self.labelFile, 'r', 'utf-8')

        self.batch_size = batch_size

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key))] + init_states
        self.provide_label = [('label', (self.batch_size, self.default_bucket_key))]


    def __iter__(self):

        featLineNum = 0
        labelLineNum = 0
        while True:
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

                line4BatchRead = line4BatchRead + 1

                # 2. str to array
                splits = line.split(',')
                lenSplits = len(splits)
                if featMaxDimLen < lenSplits:
                    featMaxDimLen = lenSplits

                item = [float(n) for n in splits]
                featBatchItems.append(item)
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

                line4BatchRead = line4BatchRead + 1

                # 2. str to array
                splits = line.split(',')
                lenSplits = len(splits)

                if labelMaxDimeLen < lenSplits:
                    labelMaxDimeLen = lenSplits

                item = [float(n) for n in splits]
                labelBatchItems.append(item)

                if line4BatchRead >= self.batch_size:
                    break
            
            # 3. feature array to np.array
            data = np.zeros((self.batch_size, featMaxDimLen))
            for i in range(len(self.batch_size)):
                data[i][:len(featBatchItems[i])] = featBatchItems[i]

            # 4. label array to np.array
            label = np.zeros((self.batch_size, labelMaxDimeLen))
            for i in range(len(self.batch_size)):
                label[i][:len(labelBatchItems[i])] = labelBatchItems[i]

            
            # 5. fill data for iterator
            data_all = [mx.nd.array(data)] + self.init_state_arrays    
            data_names = ['data'] + init_state_names

            # 6. fill label for iterator
            label_all = [mx.nd.array(label)]
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])
            yield data_batch


    def reset(self):
        pass
