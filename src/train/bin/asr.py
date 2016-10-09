# coding=utf-8
# 定义模型结构, 数据预处理.
from os import path
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 12, """batch size""")
tf.app.flags.DEFINE_string('data_dir', '/asrDataCenter/dataCenter/asr/td/vx/binaryFormat/200h/',
                           """train data dir""")

tf.app.flags.DEFINE_integer('train_file_number', 24, """训练文件的个数""")
tf.app.flags.DEFINE_string("train_file_pre",
                           "train.eesen.shuffle.",
                           """记录每个小文件的每行的最大的值的个数""")
tf.app.flags.DEFINE_string("train_maxsize_file",
                           "train.maxsize",
                           """记录每个小文件的每行的最大的值的个数""")

tf.app.flags.DEFINE_string('eval_file', 'cv.eesen.0',
                           """cv的样本文件,只有一个""")

tf.app.flags.DEFINE_string('cv_maxsize_file', "cv.maxsize",
                           """记录每个小文件的每行的最大的值的个数""")

tf.app.flags.DEFINE_integer('feature_cols', 120, """一个帧的包含的特征的个数""")
tf.app.flags.DEFINE_integer('num_layers', 5, """lstm网络的lstm的层数""")
tf.app.flags.DEFINE_integer('num_hidden', 320, """lstm网络每层的节点数""")
tf.app.flags.DEFINE_integer('num_epochs', 200, """迭代的次数""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.004, """学习率""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9, "学习率下降的百分比")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """""")


class TrainDataInfo:
  """存放训练数据的一些信息，比如一个样本的长度
  一共有多少个样本等信息
  """
  def __init__(self):
    # 一个样本所有部分占用固定最大的float32的长度
    self.example_max_length = 0
    # 一个样本的特征部分的固定占用最大float32的长度
    self.example_feature_max_length = 0
    # 一个样本的label部分的固定占用最大float32的长度
    self.example_label_max_length = 0
    # 一共有多少个样本
    self.example_number = 0
    # 所有的label的个数， 一般是70-90左右
    self.example_label_count = 0
    # 一帧包含多少个特征值
    self.feature_cols = 120


def read_data_config(config_path):
  """读取数据配置文件.

  Args:
    config_path: 配置文件路径.

  Returns:
    (一个样本最大的占多少个float32,
    特征部分占多少个float32的个数,
    label最大占多少个float32的个数,
    样本的个数,
    label的类型的个数,
    一帧数据的特征数个数)

  """
  config = list()

  with open(config_path) as config_file:
    for line in config_file:
      config.append(int(line.strip()))

  traindata_info = TrainDataInfo()
  traindata_info.example_max_length = config[0]
  traindata_info.example_feature_max_length = config[1]
  traindata_info.example_label_max_length = config[2]
  traindata_info.example_number = config[3]
  traindata_info.example_label_count = config[4]
  return traindata_info


def get_dev_data(dev_data_config):
  """获取开发集数据.

  Args:
    dev_data_config: 开发集数据配置.

  Returns:
    开发集数据.

  """
  batch_size = FLAGS.batch_size
  eval_file = FLAGS.data_dir + FLAGS.eval_file
  if not eval_file:
    raise ValueError('Please supply a eval data_file')

  if not tf.gfile.Exists(eval_file):
    raise ValueError('Failed to find file: ' + eval_file)
  file_names = [eval_file]

  filename_queue = tf.train.string_input_producer(file_names)
  record_bytes = dev_data_config.example_max_length * 4
  feature_cols = FLAGS.feature_cols

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  key, value = reader.read(filename_queue)

  value_n = tf.decode_raw(value, tf.float32)
  value_n = tf.reshape(value_n, [dev_data_config.example_max_length])

  min_queue_examples = 300

  inputs = tf.train.shuffle_batch([value_n], batch_size=batch_size,
                                  num_threads=16,
                                  capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples)

  return inputs


def distort_inputs(data_config):
  """乱序读入训练数据.

  Args:
    data_config: 数据相关的参数. 结构为:
    (一个样本最大的占多少个float32, 特征部分占多少个float32的个数,
    label最大占多少个float32的个数, 样本的个数)

  Returns:
    读取数据的op.

  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  data_dir = FLAGS.data_dir
  train_file_number = FLAGS.train_file_number
  batch_size = FLAGS.batch_size
  data_paths = list()

  for i in range(train_file_number):
    data_paths.append(path.join(data_dir, FLAGS.train_file_pre + str(i)))

  for f in data_paths:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(data_paths)
  record_bytes = data_config.example_max_length * 4
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

  # 这里是读取tf.float32的, 如果二进制写入的格式变了, 下面需要对应的修改.
  key, value = reader.read(filename_queue)
  value_n = tf.decode_raw(value, tf.float32)
  value_n = tf.reshape(value_n, [data_config.example_max_length])

  min_queue_examples = 300

  return tf.train.shuffle_batch([value_n], batch_size=batch_size,
                                num_threads=4,
                                capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
  """
  动态双向RNN实现,tensorflow的master中已经实现，1.0版本还没发布
  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: The RNN inputs.
      If time_major == False (default), this must be a tensor of shape:
        `[batch_size, max_time, input_size]`.
      If time_major == True, this must be a tensor of shape:
        `[max_time, batch_size, input_size]`.
      [batch_size, input_size].
    sequence_length: An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.

  Returns:
    A tuple (outputs, output_states) where:
      outputs: A tuple (output_fw, output_bw) containing the forward and
        the backward rnn output `Tensor`.
        If time_major == False (default),
          output_fw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_bw.output_size]`.
        If time_major == True,
          output_fw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_bw.output_size]`.
        It returns a tuple instead of a single concatenated `Tensor`, unlike
        in the `bidirectional_rnn`. If the concatenated one is preferred,
        the forward and backward outputs can be concatenated as
        `tf.concat(2, outputs)`.
      output_states: A tuple (output_state_fw, output_state_bw) containing
        the forward and the backward final states of bidirectional rnn.
  """

  if not isinstance(cell_fw, tf.nn.rnn_cell.RNNCell):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not isinstance(cell_bw, tf.nn.rnn_cell.RNNCell):
    raise TypeError("cell_bw must be an instance of RNNCell")

  with tf.variable_scope(scope or "BiRNN"):
    # Forward direction
    with tf.variable_scope("FW") as fw_scope:
      output_fw, output_state_fw = tf.nn.dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)

    # Backward direction
    if not time_major:
      time_dim = 1
      batch_dim = 0
    else:
      time_dim = 0
      batch_dim = 1

    with tf.variable_scope("BW") as bw_scope:
      inputs_reverse = tf.reverse_sequence(
          input=inputs, seq_lengths=tf.to_int64(sequence_length),
          seq_dim=time_dim, batch_dim=batch_dim)
      tmp, output_state_bw = tf.nn.dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)

  output_bw = tf.reverse_sequence(
      input=tmp, seq_lengths=tf.to_int64(sequence_length),
      seq_dim=time_dim, batch_dim=batch_dim)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)



def rnn(inputs, data_config, scope_name):
  """构建训练模型的RNN网络.

  Args:
    inputs: 数据.
    data_config: 数据相关配置. 是一个n元组.

  Returns:

  """
  batch_size = FLAGS.batch_size
  num_layers = FLAGS.num_layers
  num_hidden = FLAGS.num_hidden
  feature_cols = data_config.feature_cols

  label_count = data_config.example_label_count + 2 #label的个数+2
  num_classes = label_count

  seq_len = tf.slice(inputs, [0, 0], [batch_size, 1])
  seq_len = tf.reshape(seq_len, [batch_size])

  seq_len = tf.div(seq_len, feature_cols)
  seq_len = tf.to_int32(seq_len, name="seq_len")

  label_len = tf.slice(inputs, [0, 1], [batch_size, 1])
  label_len = tf.to_int32(tf.reshape(label_len, [batch_size]), name="label_len")

  feature_area = tf.slice(inputs, [0, 2], [batch_size, data_config.example_feature_max_length])
  feature_area = tf.reshape(feature_area, [batch_size, -1, feature_cols], name="feature_area")

  label_area = tf.slice(inputs, [0, 2 + data_config.example_feature_max_length],
                        [batch_size, data_config.example_label_max_length], name="label_area")

  with tf.device('/cpu:0'):
    cell_fw  = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    stack_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers,
                                        state_is_tuple=True)
    cell_bw  = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    stack_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers,
                                        state_is_tuple=True)

    W = tf.get_variable("weights", [num_hidden*2, num_classes],
                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                 stddev=0.1))
    b = tf.get_variable("biases", [num_classes],
                        initializer=tf.constant_initializer(0.0))
  outputs, statuses = bidirectional_dynamic_rnn(stack_fw,stack_bw, 
                        feature_area,seq_len, dtype=tf.float32)

  outputs= tf.concat(2,outputs, name="output_concat")

  # 做一个全连接映射到label_num个数的输出

  outputs = tf.reshape(outputs, [-1, num_hidden*2])


  logits = tf.add(tf.matmul(outputs, W), b, name="logits_add")
  logits = tf.reshape(logits, [batch_size, -1, num_classes])

  ctc_input = tf.transpose(logits, (1, 0, 2))

  # label的所有值
  label_value = tf.slice(label_area, [0, 0], [1, label_len[0]])
  for i in range(1, batch_size):
    v1 = tf.slice(label_area, [i, 0], [1, label_len[i]])
    label_value = tf.concat(1, [label_value, v1])

  label_value = tf.reshape(label_value, [-1])

  # [0,1,2,3,4,5,6,....,n]
  indices = [0]
  for i in range(1, data_config.example_label_max_length):
    indices = tf.concat(0, [indices, [i]])

  batch_size_array = [0]
  for i in range(1, batch_size):
    batch_size_array = tf.concat(0, [batch_size_array, [i]])

  indice_matrix = tf.tile(batch_size_array, [data_config.example_label_max_length])
  indice_matrix = tf.reshape(indice_matrix, [-1, batch_size])
  indice_matrix = tf.transpose(indice_matrix, (1, 0))

  indice1 = tf.slice(indice_matrix, [0, 0], [1, label_len[0]])
  indice1 = tf.reshape(indice1, [-1])
  indice2 = tf.slice(indices, [0], [label_len[0]])
  indice_array = tf.pack([indice1, indice2], axis=1)
  for i in range(1, batch_size):
    indice1 = tf.slice(indice_matrix, [i, 0], [1, label_len[i]])
    indice1 = tf.reshape(indice1, [-1])
    indice2 = tf.slice(indices, [0], [label_len[i]])
    temp_array = tf.pack([indice1, indice2], axis=1)
    indice_array = tf.concat(0, [indice_array, temp_array])

  sparse_shape = [batch_size, data_config.example_label_max_length]
  sparse_shape = tf.to_int64(sparse_shape)
  indice_array = tf.to_int64(indice_array)
  label_value = tf.to_int32(label_value)
  targets = tf.SparseTensor(indice_array, label_value, sparse_shape)

  return ctc_input, targets, seq_len

def loss_multi(logits, targets, seq_len):
  loss = tf.nn.ctc_loss(logits, targets, seq_len)
  cost = tf.reduce_mean(loss)
  tf.add_to_collection('losses', cost)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == "__main__":
  pass
