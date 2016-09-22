#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Kuang.Ru on 2016/8/29
from os import path

import tensorflow as tf


class Model:
  """该类抽象了ASR模型的各项基本功能.

  Attributes:
    train_file_number: 使用训练文件的数量.
    batch_size: batch大小.
    num_epochs: epoch的数量.
  """


  def __init__(self, batch_size):
    self.batch_size = batch_size


  def train(self, file_paths, num_epochs):
    """指定训练数据, 开始训练模型.

    Args:
      file_paths: 多个文件的路径列表.

    """
    self.train_file_number = len(file_paths)
    self.num_epochs = num_epochs
    raise NotImplementedError


  def test(self, audio_input):
    """传入一句话的语音数据, 产生对应的文本.

    Args:
      audio_input: 语音输入数据.

    Returns:
      语音数据对应的文本.

    """
    raise NotImplementedError


  def load(self, model_path):
    """载入模型参数.

    Args:
      model_path: 模型参数保存的路径.

    """
    raise NotImplementedError


  def distort_train_inputs(self, data_dir, data_config, num_epochs):
    """乱序读入训练数据.

    Args:
      data_dir: 数据所在的文件夹.
      data_config: 数据相关的参数. 结构为:
      (一个样本最大的占多少个float32, 特征部分占多少个float32的个数,
      label最大占多少个float32的个数, 样本的个数)

      num_epochs: 训练数据需要经过多少个epoch处理.

    Returns:
      读取数据的op.

    """
    if not data_dir:
      raise ValueError("Please supply a data_dir")

    data_dir = data_dir
    train_file_number = self.train_file_number
    batch_size = self.batch_size
    data_paths = list()

    for i in range(train_file_number):
      data_paths.append(path.join(data_dir, "train.eesen." + str(i)))

    for f in data_paths:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(data_paths, num_epochs)
    record_bytes = data_config[0] * 4
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    # 这里是读取tf.float32的, 如果二进制写入的格式变了, 下面需要对应的修改.
    key, value = reader.read(filename_queue)
    value_n = tf.decode_raw(value, tf.float32)
    value_n = tf.reshape(value_n, [data_config[0]])
    min_fraction_of_queue = 0.01
    num_examples_per_epoch = data_config[3]
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_queue)

    return tf.train.shuffle_batch([value_n], batch_size=batch_size,
                                  num_threads=4,
                                  capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples)


class TrainConfig:
  pass