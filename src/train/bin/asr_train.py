# coding=utf-8
"""
训练语音识别
特征数据有eesen的脚本先生成好， 通过本目录的: format_eesen_data.py 程序将eesen生成的文本
特征文件转成二进制的文件.

asr训练程序通过tf.FixLengthReader读取二进制文件得到特征和label
"""
import time
import logging
import os

import tensorflow as tf


import asr

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9, 'gpu占用内存比例')
tf.app.flags.DEFINE_string('train_dir', "../data/model/", '保存模型数据的文件夹')
tf.app.flags.DEFINE_integer('reload_model', 0, '是否reload之前训练好的模型')


def train():
  """训练LSTM + CTC的语音识别系统.

  """
  train_data_config = asr.read_data_config(FLAGS.train_maxsize_file)
  dev_data_config = asr.read_data_config(FLAGS.cv_maxsize_file)
  train_data = asr.distort_inputs(train_data_config)
  dev_data = asr.get_dev_data(dev_data_config)

  with tf.variable_scope("inference") as scope:
    ctc_input, train_targets, train_seq_len = asr.rnn(train_data,
                                                      train_data_config)

    scope.reuse_variables()
    dev_ctc_in, dev_targets, dev_seq_len = asr.rnn(dev_data, dev_data_config)

  train_cost, train_op = asr.build_ctc(ctc_input, train_targets, train_seq_len)
  dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in, dev_seq_len)

  edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), dev_targets,
                                   normalize=False)

  batch_error_count = tf.reduce_sum(edit_distance)
  batch_label_count = tf.shape(dev_targets.values)[0]
  init = tf.initialize_all_variables()
  local_init = tf.initialize_local_variables()
  saver = tf.train.Saver()

  gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    if FLAGS.reload_model == 1:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      saver.restore(session, ckpt.model_checkpoint_path)

      global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

      logging.info("从%s载入模型参数, global_step = %d",
                   ckpt.model_checkpoint_path, global_step)
    else:
      logging.info("Created model with fresh parameters.")
      session.run(init)
      session.run(local_init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    step = 0
    epoch = 0
    batch_size = FLAGS.batch_size
    dev_examples_num = dev_data_config[3]
    dev_num_batches_per_epoch = int(dev_examples_num / batch_size)
    train_num_examples = train_data_config[3]
    train_num_batches_per_epoch = int(train_num_examples / batch_size)

    try:
      while not coord.should_stop():
        step += 1
        start_time = time.time()
        train_cost_value, _ = session.run([train_cost, train_op])
        duration = time.time() - start_time
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        if step % 20 == 0:
          logging.info(
            'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)',
            step, train_cost_value, examples_per_sec, sec_per_batch)

        if step % train_num_batches_per_epoch == 0:
          saver.save(session, FLAGS.train_dir + "model.ckpt")
          logging.info("保存模型参数.")
          epoch += 1
          dev_error_count = 0
          dev_label_count = 0

          for batch in range(dev_num_batches_per_epoch):
            cv_error_count_value, cv_label_count = session.run(
              [batch_error_count, batch_label_count])

            dev_error_count += cv_error_count_value
            dev_label_count += cv_label_count

          dev_acc_ratio = (dev_label_count - dev_error_count) / dev_label_count

          logging.info("eval: step = %d epoch = %d eval_acc = %.3f ",
                       step, epoch, dev_acc_ratio)
    except tf.errors.OutOfRangeError:
      logging.info("训练完成.")
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    coord.join(threads)


def main(_):
  train()


if __name__ == '__main__':
  logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                      level=logging.INFO)

  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  tf.app.run()
