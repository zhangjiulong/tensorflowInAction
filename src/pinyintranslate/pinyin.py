#!/usr/bin/env python3
#coding=utf-8
"""
Created by Eric Lo on 2010-05-20.
Copyright (c) 2010 __lxneng@gmail.com__. http://lxneng.com All rights reserved.
"""
class Pinyin():
    def __init__(self, data_path='./data/Mandarin.dat'):
        self.dict = {}
        for line in open(data_path):
            k, v = line.split('\t')
            self.dict[k] = v
        self.splitter = ' '

    def get_pinyin(self, chars=u"你好吗"):
        result = []
        for char in chars:
            key = "%X" % ord(char)
            #if char is spacekey
            if key =="20":
              continue
            try:
                result.append(self.dict[key].split(" ")[0].strip()[:-1].lower())
            except:
                result.append(char)
        return self.splitter.join(result)


    def get_initials(self, char=u'你'):
        try:
            return self.dict["%X" % ord(char)].split(" ")[0][0]
        except:
            return char
    def gen_mohu_file(self, pinyin_list_path='./data/pinyin.list',output_file='./data/mohu.list'):
        py_list=[]
        fr = open(pinyin_list_path)
        for line in fr: 
            line= line.strip()
            py_list.append(line)
        fr.close()

        fr = open(pinyin_list_path)
        fw = open(output_file,'w')
        for line in fr: 
            #前后鼻音
            line= line.strip()
            if line.endswith('N')  and line+'G' in py_list:
                fw.write("%s %s\n" % (line, line+'G'))
            if line.endswith('G') and line[:-1] in py_list:
                fw.write("%s %s\n" % (line, line[:-1]))
            #卷舌音，平舌音
            if line.startswith('Z') or line.startswith('C') or line.startswith('S'):
                newline = line[0]+'H'+line[1:]
                if newline in py_list:
                    fw.write("%s %s\n" % (line, newline))
                    fw.write("%s %s\n" % (newline, line))

            # uo vs u
            if line.endswith('UO'):
                newline = line[:-1]
                if newline in py_list:
                    fw.write("%s %s\n" % (line, newline))
                    fw.write("%s %s\n" % (newline, line))

        # 其他模糊音
        fw.write("%s %s\n" %('FU','HU'))
        fw.write("%s %s\n" %('HU','FU'))
            
        fw.close()
        fr.close()

if __name__ =="__main__":
  py= Pinyin()
  print(py.get_pinyin(u"你好 啊"))
  #py.gen_mohu_file() 
