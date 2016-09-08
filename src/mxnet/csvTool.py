#coding=utf-8
import codecs
from itertools import izip

def delTooLongLine(featFile, labelFile, featFileProced, labelFileProced):
    MAX_FEAT_LEN = 200000
    MAX_LABEL_LEN = 280

    featFileProcedPtr = codecs.open(featFileProced, 'w', 'utf-8')
    labelFileProcedPtr = codecs.open(labelFileProced, 'w', 'utf-8')
    lineNum = 0

    with codecs.open(featFile, 'r', 'utf-8') as featFilePtr, codecs.open(labelFile, 'r', 'utf-8') as labelFilePtr:
        for lineFeat, lineLabel in izip(featFilePtr, labelFilePtr):
            lineNum = lineNum + 1
            if lineNum % 10000 == 0:
                print 'processing line %d'%(lineNum)

            lineFeat = lineFeat.strip()
            lineLabel = lineLabel.strip()
            
            if len(lineFeat) <= 0:
                continue

            if len(lineLabel) <= 0:
                continue

            featSplits = lineFeat.split(',')
            labelSplits = lineLabel.split(',')
            
            if len(featSplits) > MAX_FEAT_LEN:
                continue
            
            if len(labelSplits) > MAX_LABEL_LEN:
                continue
                
            featFileProcedPtr.write(lineFeat + '\n')
            labelFileProcedPtr.write(lineLabel + '\n')

    featFileProcedPtr.close()
    labelFileProcedPtr.close()

def checkLen(inFile):
    maxLen = -1
    inFilePtr = codecs.open(inFile, 'r', 'utf-8')
    lineNum = -1

    for line in inFilePtr:
        lineNum = lineNum + 1
        
        if lineNum % 1000 == 0:
            print 'processing line %d'%(lineNum)
        splits = line.split(',')

        if len(splits) > maxLen:
            maxLen = len(splits)

    print 'max len is %d'%(maxLen)

if __name__ == '__main__':
    featFile = '/asrDataCenter/dataCenter/asr/td/vx/binaryFormat/110h_traindata_txt/cv2.csv'
    labelFile = '/asrDataCenter/dataCenter/asr/td/vx/binaryFormat/110h_traindata_txt/cv2.label.csv'
    featProcedFile = './data/cv2_rm_long.csv'
    labelProcedFile = './data/cv2.label_rm_long.csv'
    
    #delTooLongLine(featFile, labelFile, featProcedFile, labelProcedFile)
    
    checkFile = './data/train2.label_rm_long.csv'
    checkLen(checkFile)
