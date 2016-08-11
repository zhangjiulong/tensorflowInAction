#coding=utf-8

import getopt, sys, os, codecs, re, jieba

def prepareDic(dicMapFile, targetDicFile, txtFile, targetTxtFile, errorResultFile, fileSeged = True):
    
    #check dic file exists.
    if not os.path.exists(dicMapFile) and os.path.isfile(dicMapFile):
        print 'the dic file not exists'
        sys.exit(1)
    # check txt file exists
    if not os.path.exists(txtFile) and os.path.isfile(txtFile):
        print 'the txt file not exists'
        sys.exit(2)

    pureEngPattern = re.compile('^[a-zA-Z]+$')


    # def dic 4 write
    targetDicFilePtr = codecs.open(targetDicFile, 'w', 'utf-8')
    targetTxtFilePtr = codecs.open(targetTxtFile, 'w', 'utf-8')
    delTxtFilePtr = codecs.open(targetTxtFile + '.del', 'w', 'utf-8')
    txtFilePtr = codecs.open(txtFile, 'r', 'utf-8')

    # read dic file
    dicMapFilePtr = codecs.open(dicMapFile, 'r', 'utf-8')
    dict = {}
    tokens = []
    lineNum = 0
    for eachline in dicMapFilePtr:
        eachline = eachline.strip()
        lineNum = lineNum + 1
        tokens = eachline.split(' ', 1)
        if len(tokens) < 2 :
            print 'probabily dic error on line ' + str(lineNum)
            continue
        
        # add word to dict
        dict[tokens[0].strip()] = tokens[1].strip().upper()
        
    dicMapFilePtr.close()
    
    lineNum = 0
    usedDict = {}
    engNotInDic = {}
    hanNotInDic = {}
    for line in txtFilePtr:
        lineNum = lineNum + 1
        line = line.strip()
        line = line.replace('</s>', '')
        line = line.replace('<s>', '')
        
        # add by zhangjl on 20160705 processing unsegmented line.
        if not fileSeged:
            line = " ".join(jieba.cut(line, cut_all = False))
            line = line.strip()
        
        words = []
        phonetics = []
        
        wordsinline = line.split()
        need2write = True
        for wordinline in wordsinline:
            if dict.has_key(wordinline):
                phonetic = dict[wordinline]
                phonetics.append(phonetic)
                words.append(wordinline)
                usedDict[wordinline] = phonetic
            else:
        
                match = pureEngPattern.match(wordinline)
                if match:
                    print 'eng word may not in the dic [%s] line num is [%d]'%(wordinline, lineNum)
                    engNotInDic[wordinline] = 1
                    need2write = False
                    continue
                else:
                    for subword in wordinline:
                        if dict.has_key(subword):
                            phonetic = dict[subword]
                            phonetics.append(phonetic)
                            words.append(subword)
                            usedDict[subword] = phonetic
                        else:
                            
                            hanNotInDic[subword] = 1
                            if subword >= u'/u4e00' and subword <= u'/u9fa5':
                                need2write = False
                                print 'han no [%s] line is [%d]'%(subword, lineNum) 
                            else:
                                print 'no phnetic for word [%s] line is [%d]'%(subword, lineNum) 
                                pass
    
        line2write = ' '.join(words).strip()

        if need2write:
            targetTxtFilePtr.write(line2write.strip() + '\n')            
        else:
            delTxtFilePtr.write(line2write.strip() + '\n')

        
    for key in usedDict:
        targetDicFilePtr.write(key)
        targetDicFilePtr.write(' ')
        targetDicFilePtr.write(usedDict[key].strip())
        targetDicFilePtr.write('\n')
        
    errorResultFilePtr = codecs.open(errorResultFile, 'w', 'utf-8')
    
    for key in engNotInDic:
        errorResultFilePtr.write(key + '\n')

    for key in hanNotInDic:
        errorResultFilePtr.write(key + '\n')


if __name__ == '__main__':
    #dicMapFile = '/home/zhangjl/dataCenter/lm/td/zh_broadcastnews_utf8.dic'
    #targetDicFile = '/home/zhangjl/dataCenter/lm/td/td.dic'
    #txtFile = '/home/zhangjl/dataCenter/lm/td/td.txt'
    #targetTxtFile = '/home/zhangjl/dataCenter/lm/td/td_indic.txt'
    #errorResultFile = '/home/zhangjl/dataCenter/lm/td/tdresult.txt'
    #prepareDic(dicMapFile,targetDicFile, txtFile, targetTxtFile, errorResultFile)
                            

    dicMapFile = '/asrDataCenter/dataCenter/lm/td/zh_broadcastnews_utf8.dic'
    txtFile = '/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stmReArrange/stmSeg5/clear4LM.txt'
    targetDicFile = '/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stmReArrange/stmSeg5/clearedDct4LMGened.dct'
    targetTxtFile = '/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stmReArrange/stmSeg5/clearedTxt4LMGened.txt'
    errorResultFile = '/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stmReArrange/stmSeg5/errorResult.txt'
    prepareDic(dicMapFile,targetDicFile, txtFile, targetTxtFile, errorResultFile, fileSeged = True)
    
