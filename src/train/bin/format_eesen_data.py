#!/usr/bin/env python
# coding=utf-8

import os

import codecs
from struct import *

'''
  将eesen的特征文件归一到一个文件中， 方便tensorflow或者其他的框架读取
  输出后的文件是二进制格式：
'''

def format_data():
  TRAINFEATUREFILE="./eesen/exp/train_phn_l2_c120/train2.ark"
  TRAINLABELFILE="./eesen/exp/train_phn_l2_c120/labels.tr"
  #TRAINFEATUREFILE="../data/bak/timit/train_phn_l2_c120/train2.ark"
  #TRAINLABELFILE="../data/bak/timit/train_phn_l2_c120/labels.tr"
  
  CVFEATUREFILE="./eesen/exp/train_phn_l2_c120/cv2.ark"
  CVLABELFILE="./eesen/exp/train_phn_l2_c120/labels.cv"
  #CVFEATUREFILE="../data/bak/timit/train_phn_l2_c120/cv2.ark"
  #CVLABELFILE="../data/bak/timit/train_phn_l2_c120/labels.cv"

  #训练数据里包含的lable的个数
  LABEL_MAX_FILE = "./eesen/exp/train_phn_l2_c120/tr.lable.count.max" 

  format_eesen_data(TRAINFEATUREFILE, TRAINLABELFILE, LABEL_MAX_FILE, "../data/traindata/train.eesen","../data/traindata/train.maxsize", 24)
  format_eesen_data(CVFEATUREFILE, CVLABELFILE, LABEL_MAX_FILE, "../data/traindata/cv.eesen", "../data/traindata/cv.maxsize",1)
  pass


def format_eesen_data(ark_file, label_file, label_max_file, output_file_prefix, max_size_output, output_file_number=1):

  feature_number = 120

  #最大的特征的个数， 超出的过滤掉
  config_max_feature_size = 100000
 
  print("start format eesen feature data, %s...." % ark_file)

  if not os.path.exists(ark_file):
    print("%s not exist" % ark_file)
    return -1
  if not os.path.exists(label_file):
    print("%s not exist" % label_file)
    return -1

  label_dict={}

  #一行最多包含多少个值， 不足的用0补齐
  max_values_per_line = []

  max_line_number = 0

  #读取的的文件，样本的包含的特征数量，已经按照从小到大排序好了。 
  #先读取label的数据到内存， 后读取feature的数据去匹配
  with codecs.open(label_file, 'r', 'utf-8') as label_fr:
    while True:
      line = label_fr.readline()
      if line =='':
        break
      words = line.strip().split(' ')
      words_number = len(words)
      if words_number >=2:
        label_dict[words[0]] = words[1:]


  # 样本个数
  example_number = 0

  #最大的特征的个数
  max_feature_size = 0

  #最大的label的个数
  max_label_size = 0

  # 总共一行的最大的size 2+ max_feature_size + max_label_size
  max_total_size = 0

  # 分成10份，计算每一份的max_values_per_line
  with codecs.open(ark_file, 'r', 'utf-8') as ark_fr:
    feature_size= 0
    example_id = 0
    while True:
      line = ark_fr.readline()
      if line=='':
        max_feature_size = feature_size
        break
      words = line.split()
      if len(words)==2 and words[1]=='[':
        key = words[0]
        label_value = label_dict[key]
        feature_size = 0
        continue
      if len(words) == feature_number:
        feature_size = feature_size + feature_number
        continue
      if len(words) == feature_number+1 and words[-1]==']':
        feature_size = feature_size + feature_number
        label_size   = len(label_value)
        
        # 如果帧数除以label的个数小于3，说明在很短的时间里说了很多话，
        # 基本上不太可能，可能这条记录有问题，就去掉
        if (feature_size/feature_number)/label_size < 3 : 
          print("too short", key, feature_size, label_size)
          continue
        if label_size > 150:
          continue
        if feature_size/feature_number < label_size:
          print("error", key, feature_size, label_size)
          continue
 
        if label_size > max_label_size:
          max_label_size = label_size

        example_number = example_number + 1
        
        if feature_size > config_max_feature_size:
          max_feature_size = feature_size
          break


  #每个小文件的保存的样本的个数
  example_number_per_file = int(example_number / output_file_number)


  #加2 是因为前面2个数字保存feature的个数和label的个数
  max_total_size = max_feature_size + max_label_size + 2

  #读取lable的个数
  label_fr = codecs.open(label_max_file, 'r','utf-8')
  label_max = label_fr.readline()
  label_fr.close()
  label_max = int(label_max)

  print("the size info:")
  size_fw = codecs.open(max_size_output, 'w', 'utf-8')
  size_fw.write(str(max_total_size)+'\n')
  size_fw.write(str(max_feature_size)+'\n')
  size_fw.write(str(max_label_size)+'\n')
  size_fw.write(str(example_number)+'\n')
  size_fw.write(str(label_max)+'\n')
  print(max_total_size)
  print(max_feature_size)
  print(max_label_size)
  print(example_number)
  print(label_max)
  size_fw.close()

  #先把一行中所有的特征和label都读出来， 计算出来有多少个特征和label的个数， 然后用二进制的方式写入到文件
  # 前面两个数字表示特征的个数和label的个数

  #分散写到各个文件里去, 先把所有文件打开
  fw=[]
  for i in range(output_file_number+1):
    outfile_name = output_file_prefix +"."+str(i)
    f = open(outfile_name, 'wb')
    fw.append(f)

  k=0
  output_fw = fw[k]
  writed_lines = 0  # 已经在一个文件里写了多少样本的数据了

  with codecs.open(ark_file, 'r', 'utf-8') as ark_fr:
    feature_str = ""
    while True:
      line = ark_fr.readline().strip()
      if line=='':
        break
      words = line.split()
      # 如果是2个字段的方式, e.g.: fcdr1_sx70-0000000-0000091  [
      # 表示的是特征的开头
      if len(words)==2 and words[1]=='[':
        feature_str = ""
        key = words[0]
        label_value = label_dict[key]
        continue
      if len(words) == feature_number:
         feature_str = feature_str + " " + line
         continue
      if len(words) == feature_number+1 and words[-1]==']':
        feature_str  = feature_str + " " + line[:-1]
          
        feature_list = feature_str.strip().split()
        label_list   = label_value

        feature_size = len(feature_list)
        label_size   = len(label_list)


        if (feature_size/feature_number)/label_size < 3 :
          continue
        if label_size > 150:
          continue    
        if feature_size/feature_number < label_size:
          continue

        output_fw.write(pack("f", feature_size))
        output_fw.write(pack("f", label_size))

        for i in range(feature_size):
          output_fw.write(pack("f", float(feature_list[i])))
        blank_number = max_feature_size - feature_size
        if blank_number > 0:
          for i in range(blank_number):
            output_fw.write(pack("f", 0))

        for i in range(label_size):
          output_fw.write(pack("f", float(label_list[i])))

        blank_number  = max_label_size - label_size

        if blank_number > 0:
          for i in range(blank_number):
            output_fw.write(pack("f", 0))

        writed_lines = writed_lines + 1

        if writed_lines >= example_number_per_file:
          writed_lines = 0
          k = k + 1
          output_fw = fw[k]
  
        feature_str = ""

        if feature_size > config_max_feature_size:
          break

  # 关闭文件        
  for f in fw:
    f.close()

  
  print("successed process eesen feature data")


  # 是否打乱文件, 因为虽然分成了n个文件， 但是每个文件里面的长度都是从小到大的。
  # 这里每次取出n个文件的第一个记录， 写到新的文件里
  if True:
    fr=[]
    for i in range(output_file_number+1):
      readfile_name = output_file_prefix +"."+str(i)
      f = open(readfile_name, 'rb')
      fr.append(f)

    fw=[]
    for i in range(output_file_number+1):
      outfile_name = output_file_prefix +".shuffle."+str(i)
      f = open(outfile_name, 'wb')
      fw.append(f)

    shuffled_example = 0
    writefile_index = 0
    while shuffled_example < example_number:
      for i in range(output_file_number+1):
        readdata = fr[i].read(max_total_size * 4)  # 4 为float大小
        fw[writefile_index].write(readdata)

        shuffled_example += 1
        if shuffled_example % example_number_per_file == 0:
          fw[writefile_index].flush()
          writefile_index += 1
    fw[writefile_index].flush()
    for f in fw:
      f.close()
    for f in fr:
      f.close()



if __name__ == "__main__":
  format_data()

