#coding=utf-8
'''
分布式的方式训练语音识别的声学模型
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
from os import path
import time

import tensorflow as tf
import asr

import logging

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction',0.9,'gpu占用内存比例')
tf.app.flags.DEFINE_string('model_dir',"../data/model_distribute/",'保存模型数据的文件夹')
tf.app.flags.DEFINE_integer('num_epochs_per_decay',10,'多少个epoch之后学习率下降')
tf.app.flags.DEFINE_string('cuda_visible_devices',"0",'使用第几个GPU')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")


def train():
  """训练LSTM + CTC的语音识别系统.
  
  """

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

  
  issync = FLAGS.issync
  
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    data_dir = FLAGS.data_dir
    cv_maxsize_file = path.join(data_dir , FLAGS.cv_maxsize_file)
    train_maxsize_file = path.join(data_dir , FLAGS.train_maxsize_file)

    train_data_config = asr.read_data_config(train_maxsize_file)
    dev_data_config = asr.read_data_config(cv_maxsize_file)
    train_data = asr.distort_inputs(train_data_config)
    dev_data = asr.get_dev_data(dev_data_config)

    batch_size = FLAGS.batch_size
    dev_examples_num = dev_data_config.example_number
    dev_num_batches_per_epoch = int(dev_examples_num / batch_size)
    train_num_examples = train_data_config.example_number
    train_num_batches_per_epoch = int(train_num_examples / batch_size)
  
    # 多少个step之后， 学习率下降
    decay_steps = int(train_num_batches_per_epoch * FLAGS.num_epochs_per_decay)
  
    initial_learning_rate = FLAGS.initial_learning_rate
    learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
    
    with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
      global_step = tf.Variable(0, name='global_step', trainable=False)
      #lr = tf.train.exponential_decay(initial_learning_rate, global_step,decay_steps, learning_rate_decay_factor, staircase=True)
      #optimizer = tf.train.AdamOptimizer(lr)

      with tf.variable_scope("inference") as scope:
        ctc_input, train_targets, train_seq_len = asr.rnn(train_data,
                                                      train_data_config,"train")

        scope.reuse_variables()
        dev_ctc_in, dev_targets, dev_seq_len = asr.rnn(dev_data, dev_data_config,"cv")

      example_losses = tf.nn.ctc_loss(ctc_input, train_targets, train_seq_len)
      train_cost = tf.reduce_mean(example_losses)

      optimizer = tf.train.AdamOptimizer(initial_learning_rate)
      if issync==1:
        rep_op = tf.train.SyncReplicasOptimizer(optimizer, 
                            replicas_to_aggregate=len(worker_hosts),
                            replica_id=FLAGS.task_index, 
                            total_num_replicas=len(worker_hosts),
                            use_locking=True )
        train_op = rep_op.minimize(train_cost, global_step=global_step)
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
      else:
        train_op  = optimizer.minimize(train_cost, global_step=global_step)

      tf.scalar_summary("train_cost", train_cost)

      dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in, dev_seq_len)

      edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), dev_targets,
                                   normalize=False)

      batch_error_count = tf.reduce_sum(edit_distance)
      batch_label_count = tf.shape(dev_targets.values)[0]
      init_op = tf.initialize_all_variables()
      local_init = tf.initialize_local_variables()
      saver = tf.train.Saver()
      summary_op     = tf.merge_all_summaries()

      gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir=FLAGS.model_dir,
                            init_op=init_op,
                            local_init_op=local_init,
                            summary_op=summary_op,
                            saver=saver,
                            global_step=global_step,
                            save_model_secs=600)    	  
      with sv.prepare_or_wait_for_session(server.target, 
                     config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        if FLAGS.task_index == 0 and issync == 1:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_token_op)

        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir, sess.graph)

        step = 0
        valid_step = 0
        epoch = 0
        while not sv.should_stop() and step < 100000000:

          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(sess=sess, coord=coord)

          try:
            while not coord.should_stop():
              train_cost_value, _,step = sess.run([train_cost, train_op,global_step])

              if step % 50 == 0:
                logging.info("step: %d, loss: %f" %(step, train_cost_value))

              # 当跑了steps_to_validate个step，并且是主的worker节点的时候， 评估下数据
              # 因为是分布式的，各个节点分配了不同的step，所以不能用 % 是否等于0的方法
              if step - valid_step > train_num_batches_per_epoch  and FLAGS.task_index == 0:
                valid_step = step
                dev_error_count = 0
                dev_label_count = 0

                for batch in range(dev_num_batches_per_epoch):
                  cv_error_count_value, cv_label_count = sess.run(
                      [batch_error_count, batch_label_count])

                  dev_error_count += cv_error_count_value
                  dev_label_count += cv_label_count

                dev_acc_ratio = (dev_label_count - dev_error_count) / dev_label_count

                logging.info("epoch: %d eval: step = %d eval_acc = %.3f ",epoch, 
                           step,  dev_acc_ratio)
                epoch += 1

          except tf.errors.OutOfRangeError:
            print("Done training after reading all data")
          finally:
            coord.request_stop()

          # Wait for threads to exit
          coord.join(threads)


def main(_):
  train()


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',  
                    datefmt='%a, %d %b %Y %H:%M:%S',  
                    filename='./out.log',  
                    filemode='a')
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
  tf.app.run()










