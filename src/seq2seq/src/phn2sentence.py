'''
音素到句子的翻译模型训练
'''
#coding=utf-8
import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

import gc
import pinyin

tf.app.flags.DEFINE_float("gpu_use_ratio", 0.9,
                          "使用gpu的显存比例.")
tf.app.flags.DEFINE_float("learning_rate", 0.3, "学习率")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95,
                          "学习率下降比例")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,"")
tf.app.flags.DEFINE_integer("size", 1024, "每一层的节点数.")
tf.app.flags.DEFINE_integer("num_layers", 3, "网络结构的层数.")
tf.app.flags.DEFINE_integer("phn_vocab_size", 200, "音素的最大个数.")
tf.app.flags.DEFINE_integer("char_vocab_size", 4000, "汉字的最大个数.")
tf.app.flags.DEFINE_string("data_dir", "data/", "数据目录，包含训练数据")
tf.app.flags.DEFINE_string("model_dir", "../model/", "模型保存的目录.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 6000,
                            "多少个step之后保存模型.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")


FLAGS = tf.app.flags.FLAGS


_buckets = [(2, 1),(4, 2),(10, 5), (16, 8),(20,10),(30,15),(40,20),(50,25),(60,30),(80,40),(100,50)]

class BucketBuffer(object):
  '''
    use to process bucket read from file
  '''
  def __init__(self, buckets,batch_size):
    self.buckets = buckets
    self.batch_size = batch_size

    self.total_read_counter = 0
    self.data_set = [[] for _ in buckets]


  def read_data(self, source_file, target_file):
    source, target = source_file.readline(), target_file.readline()
    if not source or not target:
      #如果一开始读就已经到了文件末尾了， 就从头开始
      source_file.seek(0,0)
      target_file.seek(0,0)
      source, target = source_file.readline(), target_file.readline()

    while source and target:
      self.total_read_counter+=1
      if self.total_read_counter % 10000 ==0:
        print("  reading data line %d" % self.total_read_counter)
        sys.stdout.flush()

      source_ids = [int(x) for x in source.split()]
      target_ids = [int(x) for x in target.split()]
      target_ids.append(data_utils.EOS_ID)
      for bucket_id, (source_size, target_size) in enumerate(self.buckets):
        if len(source_ids) < source_size and len(target_ids) < target_size:
          self.data_set[bucket_id].append([source_ids, target_ids])
          break
      if len(self.data_set[bucket_id]) == self.batch_size:
        patch_data = self.data_set[bucket_id]
        self.data_set[bucket_id] = []
        return patch_data, bucket_id
      source, target = source_file.readline(), target_file.readline()

      #如果文件读完了， seek到最开头
      if not source or not target:
        source_file.seek(0,0)
        target_file.seek(0,0)
        source, target = source_file.readline(), target_file.readline()

def read_data(source_path, target_path):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.phn_vocab_size, FLAGS.char_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  # 准备数据
  print("Preparing data in %s" % FLAGS.data_dir)
  phn_train, char_train, phn_dev, char_dev, _, _ = data_utils.prepare_data(
      FLAGS.data_dir, FLAGS.phn_vocab_size, FLAGS.char_vocab_size)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_use_ratio)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # 创建模型.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    print ("Reading development and training data to memery.")
    dev_set = read_data(phn_dev, char_dev)

    # 训练.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    bucket_reader = BucketBuffer(_buckets, FLAGS.batch_size)
    source_file = open(en_train, "r")
    target_file = open(fr_train, "r")

    while True:
      start_time = time.time()
      patch_data, bucket_id = bucket_reader.read_data(source_file, target_file)
      encoder_inputs, decoder_inputs, target_weights = model.get_batch2(
          patch_data, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1
      
      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')

        current_time = time.time()
        ltime = time.localtime(current_time)
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", ltime)  
        print ("%s global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (timestr, model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
                         
        # 如果loss比最后三次的loss中最大的还要大的话， 减少学习率.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        step_time, loss = 0.0, 0.0

        if current_step % (FLAGS.steps_per_checkpoint*2) ==0:
          checkpoint_path = os.path.join(FLAGS.model_dir, "translate.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        if current_step % (FLAGS.steps_per_checkpoint*2) ==0:
          for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                dev_set, bucket_id)
            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()
    source_file.close()
    target_file.close()

def decode():
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    phn_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.phn" % FLAGS.phn_vocab_size)
    char_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.char" % FLAGS.char_vocab_size)
    phn_vocab, _ = data_utils.initialize_vocabulary(phn_vocab_path)
    _, rev_char_vocab = data_utils.initialize_vocabulary(char_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      print(sentence)
      token_ids = data_utils.sentence_to_token_ids(sentence, phn_vocab)
      # 判断属于哪个bucket
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)

      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

      print(" ".join([tf.compat.as_str(rev_char_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def main(_):
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
