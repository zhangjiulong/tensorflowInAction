#coding=utf-8
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib
import numpy
from tensorflow.python.platform import gfile

import codecs
import pinyin
import time

# Special vocabulary symbols - we always put them at the start.
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


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    #with gfile.GFile(data_path, mode="rb") as f:
    with open(data_path, "r", buffering=1024) as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(str(_DIGIT_RE), b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      #with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      with open(vocabulary_path, "w",encoding='utf-8',buffering=1024) as vocab_file:
        for w in vocab_list:
          vocab_file.write(str(w) + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
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


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
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
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def cut_sentence(line):
  """
    将一行内容切分成句子. 将包含数字的句子过滤掉先
  """
  punt_list = set("，,!?:;~。！？：；～、.-\"\t　 …—")
  
  sentences = list()
  start = 0
  i = 0  # 记录每个字符的位置
  for char in line:
    if char in punt_list:
      seg_line = line[start:i ]
      match = re.findall(r'([0-9a-zA-Z【】（）)(\[\]/]+?)',seg_line, re.M|re.U)
      if len(match) ==0:
        sentences.append(seg_line)
      start = i + 1  # start标记到下一句的开头
      i += 1
    else:
      i += 1  # 若不是标点符号，则字符位置继续前移
  if start < len(line):
    sentences.append(line[start:])  # 这是为了处理文本末尾没有标点符号的情况
  return sentences

#处理中文
def pre_process_traindata(train_data_dir, train_data_filename, mohu_switch=0):
  input_file = train_data_dir +'/'+train_data_filename
  correct_output_file = train_data_dir +'/'+train_data_filename+'.correct'
  error_output_file = train_data_dir +'/'+train_data_filename+'.error'

  # 读入模糊音对
  mohu_list_file = "./data/mohu.list"
  mohu_dict = {}
  with open(mohu_list_file, 'r') as mohu_fr:
    for line in mohu_fr:
      line = line.lower()
      mohu_pair = line.split()
      if mohu_pair[0] in mohu_dict:
        mohu_dict[mohu_pair[0]] = mohu_dict[mohu_pair[0]] + " " + mohu_pair[1]
      else:
        mohu_dict[mohu_pair[0]] = mohu_pair[1]



  if os.path.exists(correct_output_file):
    print("file %s already exists" % correct_output_file)
    return
  if os.path.exists(error_output_file):
    print("file %s already exists" % error_output_file)
    return

  #生成拼音作为src的训练数据
  py = pinyin.Pinyin()
  with open(input_file, 'r',encoding='utf-8', buffering=1024) as fr:
    with open(correct_output_file, 'w',buffering=1024) as fw:
      with open(error_output_file, 'w', buffering=1024) as pyfw:
        for line in fr:
          sentences = cut_sentence(line)
          for s in sentences:
            s = re.sub("[’‘》《』『“”]",'',s)
          
            outline = ' '.join(s.strip())
            if len(outline)<2:
              continue
            fw.write(outline+'\n')
            py_line = py.get_pinyin(outline.strip())
            pyfw.write(py_line+'\n')






            # 添加模糊的音进入训练数据
            if mohu_switch==1:
              py_words = py_line.split()
              i = 0
              while i<len(py_words):
                if py_words[i] not in mohu_dict:
                  i+=1
                  continue
                # 如果连续2个字都有模糊音
                if i< (len(py_words)-1) and py_words[i+1] in mohu_dict:
                  first_mohu_str    = mohu_dict[py_words[i]]
                  second_mohu_str   = mohu_dict[py_words[i+1]]
                  first_mohu_array = first_mohu_str.split()
                  second_mohu_array = second_mohu_str.split()
                  for m in first_mohu_array:
                    for n in second_mohu_array:
                      new_py_words = py_words[:i] + [m] + [n] + py_words[i+2:]
                      new_py_line = ' '.join(new_py_words)
                      fw.write(outline+'\n')
                      pyfw.write(new_py_line+'\n')
                  i+=1
                  continue
                #只有当前字有模糊音
                first_mohu_str    = mohu_dict[py_words[i]]
                first_mohu_array = first_mohu_str.split()
                for m in first_mohu_array:
                  new_py_words = py_words[:i] + [m] +  py_words[i+1:]
                  new_py_line = ' '.join(new_py_words)
                  fw.write(outline+'\n')
                  pyfw.write(new_py_line+'\n')
                  i+=1
              



def prepare_wmt_data(data_dir, error_vocabulary_size, correct_vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  #train_path = get_wmt_enfr_train_set(data_dir)
  #dev_path = get_wmt_enfr_dev_set(data_dir)
  pre_process_traindata('./data/','traindata')
  pre_process_traindata('./data/','devdata')

  train_path = './data/traindata'
  dev_path = './data/devdata'

  # Create vocabularies of the appropriate sizes.
  correct_vocab_path = os.path.join(data_dir, "vocab%d.fr" % correct_vocabulary_size)
  error_vocab_path = os.path.join(data_dir, "vocab%d.en" % error_vocabulary_size)
  # 生成法语英语的词汇表， 根据原始原料生成
  create_vocabulary(correct_vocab_path, train_path + ".correct", correct_vocabulary_size, tokenizer)
  create_vocabulary(error_vocab_path, train_path + ".error", error_vocabulary_size, tokenizer)

  #后面就是把原始原料变成数字的表示， 每个单词改成数字
  # Create token ids for the training data.
  correct_train_ids_path = train_path + (".ids%d.correct" % correct_vocabulary_size)
  error_train_ids_path = train_path + (".ids%d.error" % error_vocabulary_size)
  data_to_token_ids(train_path + ".correct", correct_train_ids_path, correct_vocab_path, tokenizer)
  data_to_token_ids(train_path + ".error", error_train_ids_path, error_vocab_path, tokenizer)

  # Create token ids for the development data.
  correct_dev_ids_path = dev_path + (".ids%d.correct" % correct_vocabulary_size)
  error_dev_ids_path = dev_path + (".ids%d.error" % error_vocabulary_size)
  data_to_token_ids(dev_path + ".correct", correct_dev_ids_path, correct_vocab_path, tokenizer)
  data_to_token_ids(dev_path + ".error", error_dev_ids_path, error_vocab_path, tokenizer)

  return (error_train_ids_path, correct_train_ids_path,
          error_dev_ids_path, correct_dev_ids_path,
          error_vocab_path, correct_vocab_path)

if __name__=="__main__":
  a="说法 是飞 洒发"
  print(basic_tokenizer(a))
  a="as dd dd ew"
  print(basic_tokenizer(a))
  create_vocabulary('./data/vocab1000.en',  "./data/traindata.error", 1000, None)
