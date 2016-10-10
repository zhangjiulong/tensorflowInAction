#coding=utf-8
import gzip
import os
import re
import tarfile

from six.moves import urllib
import numpy
import tensorflow as tf
from tensorflow.python.platform import gfile

import codecs

import time
import jieba

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('phn_dict_file', '/asrDataCenter/dataCenter/lm/td/zh_broadcastnews_utf8.dic',
                           """词和字到音素的对应关系文件""")

# 特殊词汇符号 - 习惯性放在词汇表最前面
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")



def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
  #for space_separated_fragment in sentence.strip():
    #words.extend(re.split(' ', space_separated_fragment))
    words.append(space_separated_fragment)
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
  """创建词汇表

  Args:
    vocabulary_path: 将要生成的词汇表的路径
    data_path: 数据文件，用来创建词汇表的数据文件
    max_vocabulary_size: 词汇表最大词汇个数
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path, "r", buffering=1024) as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = basic_tokenizer(line)
        for w in tokens:
          word = w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with open(vocabulary_path, "w",encoding='utf-8',buffering=1024) as vocab_file:
        for w in vocab_list:
          vocab_file.write(str(w) + "\n")


def initialize_vocabulary(vocabulary_path):
  """根据词汇表文件，生成dict
  Args:
    vocabulary_path: 词汇表文件的路径
  Returns:
    a pair: the vocabulary (字符到整数的dict), and
    the reversed vocabulary (一个list，字符的list).
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with codecs.open(vocabulary_path, "r",'utf-8') as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary):
  """把一个句子转换成token

  Args:
    sentence: 句子
    vocabulary: 词汇字典， key是字， value是数字
 
  Returns:
    对应句子的整数的token表示
  """

  words = basic_tokenizer(sentence)
  return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      normalize_digits=True):
  """把文件变成token的表示

  Args:
    data_path: 数据文件，输入文件
    target_path: 输出文件， 文件内容变成数字代替的形式
    vocabulary_path: 词汇表文件
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with open(data_path, "r",encoding='utf-8', buffering=1024) as data_file:
      with open(target_path, "w",encoding='utf-8', buffering=1024) as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def filter_sentence(line):
  """
  将一行内容切分成句子. 将包含数字的句子过滤掉先
  Args:
    line:
  输入一行文本，其中可能包含各种符号
  Returns:
  去掉了各种符号的文本数组， 可能将一句很长的文本，变成多个在数组中的元素
  并且将其中的一些字符过滤掉
  """
  punt_list = set("，,!?:;~。！？：；～、.-\"\t　 …—")

  jump_char = r'([0-9a-zA-Z【】（）)(\[\]/=○\+]+?)'
  replace_char = r'[）（%_■□·’‘》《』『“”]'

  sentences = list()
  start = 0
  i = 0  # 记录每个字符的位置
  for char in line:
    # 分句
    if char in punt_list:
      seg_line = line[start:i]
      #包含jump_char中的字符的，跳过
      match = re.findall(jump_char, seg_line, re.M | re.U)
      if len(match) == 0:
        replace_reg = re.compile(replace_char)
        seg_line = replace_reg.sub('', seg_line)
        if len(seg_line) >= 2:
          sentences.append(seg_line)
      start = i + 1  # start标记到下一句的开头
      i += 1
    else:
      i += 1  # 若不是标点符号，则字符位置继续前移
  if start < len(line):
    seg_line = line[start:]  # 这是为了处理文本末尾没有标点符号的情况
    match = re.findall(jump_char, seg_line, re.M | re.U)
    if len(match) == 0:
      replace_reg = re.compile(replace_char)
      seg_line = replace_reg.sub('', seg_line)
      if len(seg_line) >= 2:
        sentences.append(seg_line)
  return sentences

def get_phn(sentence, phn_dict):
  '''
  得到一个句子中字的音素,先将句子分词，分词后的词去词典里去查，
  如果能查到，就用词的音素，查不到就用单个字的音素
  Args:
    sentence: 句子
    phn_dict：音素的词典， key是字或者词， value是音素值的字符串
  Returns:
    返回输入句子的音素，字符串形式，空格分开
  '''
  phn_string = ""
  segment = jieba.cut(sentence)
  for word in segment:
    if word in phn_dict:
      phn_string = phn_string + " " + ' '.join(phn_dict[word])
    else:  
      char_list = word.strip()
      for char in char_list:
        if char not in phn_dict:
          print(char)
  return phn_string.strip()


def pre_process_traindata(train_data_dir, train_data_filename):
  '''
  将中文文本文件，过滤特殊字符的内容，生成两个文件，
  一个是原来的文件的汉字用空格分开
  一个是对应的音素的文件
  Args:
    train_data_dir:  输入文件的目录
    train_data_filename： 输入文件的文件名
  Returns:
  '''
  input_file = train_data_dir +'/'+train_data_filename
  char_output_file = train_data_dir +'/'+train_data_filename+'.char'
  phn_output_file = train_data_dir +'/'+train_data_filename+'.phn'

  if os.path.exists(char_output_file):
    print("file %s already exists" % char_output_file)
    return
  if os.path.exists(phn_output_file):
    print("file %s already exists" % phn_output_file)
    return

  # 读入包含词和字的汉字到音素的转换的文件
  phn_file = FLAGS.phn_dict_file
  phn_dict = {}
  with open(phn_file, 'r') as phn_fr:
    for line in phn_fr:
      line_list = line.split()
      key = line_list[0]
      phn_dict[key] = line_list[1:]

  #生成拼音作为src的训练数据
  with open(input_file, 'r',encoding='utf-8', buffering=1024) as fr:
    with open(char_output_file, 'w',buffering=1024) as char_fw:
      with open(phn_output_file, 'w', buffering=1024) as phn_fw:
        for line in fr:
          sentences = filter_sentence(line)
          for s in sentences:
            phn_string  = get_phn(s, phn_dict)
            char_string = ' '.join(s.strip())


def prepare_data(data_dir, phn_vocabulary_size, char_vocabulary_size):
  """准备数据，初始化数据，将原始的文本数据，将其中的杂质去掉， 
  生成音素文件，汉字文件，生成音素文件的token，汉字文件的token

  Args:
    data_dir: 数据目录
    phn_vocabulary_size: 音素词典的包含音素的个数.
    char_vocabulary_size: 汉字词典包含汉字的个数.

  Returns:
    包含6个元素的元组:
      (1) 音素训练集token的dataset,
      (2) 汉字训练集token的dataset,
      (3) 音素开发集token的dataset,
      (4) 汉字开发集token的dataset,
      (5) 音素词典文件,
      (6) 汉字词典文件.
  """
  
  # 处理原始数据，生成音素文件
  pre_process_traindata('../data/','traindata')

if __name__=="__main__":
  prepare_data("../data/", 200, 4000)
