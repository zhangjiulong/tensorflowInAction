#coding=utf-8

import pinyin
import os


def delete_file_folder(src):  
  if os.path.isfile(src):  
    try:  
      os.remove(src)  
    except:  
      pass 
  elif os.path.isdir(src):  
    for item in os.listdir(src):  
      itemsrc=os.path.join(src,item)  
      delete_file_folder(itemsrc)  
    try:  
      os.rmdir(src)  
    except:  
      pass 

def convert2pinyin(input_dir, output_dir):
  # 文件夹是否存在
  if os.path.exists(output_dir):
    delete_file_folder(output_dir)

  os.makedirs(output_dir)

  for file_name in os.listdir(input_dir):
    input_file = input_dir+"/"+file_name
    output_file = output_dir+"/"+file_name
    with open(input_file, 'r') as fr:
      with open(output_file,'w') as fw:
        for line in fr:
          line_py = py.get_pinyin(line)
          fw.write(line_py)

if __name__ == "__main__":
  py = pinyin.Pinyin()

  input_dir="./output"
  output_dir="./pinyin_output"
  
  convert2pinyin(input_dir, output_dir)
