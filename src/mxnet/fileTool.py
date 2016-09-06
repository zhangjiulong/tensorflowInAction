#coding=utf-8
import os 
import sys
import glob
import re
import codecs

def getPureFileNameWithOutExt(fileName):
    tmpLastPart = os.path.split(fileName)[1]
    ret = os.path.splitext(tmpLastPart)[0]
    return ret

def getPureFileName(fileName):
    tmpLastPart = os.path.split(fileName)[1]
    ret = tmpLastPart
    return ret


def listDir(pathWithPattern):
    return glob.glob(pathWithPattern)

def transFileLower(inFile, outFile):
    f = open(inFile)
    of = open(outFile, 'w')
    while 1:
        line = f.readline()
        if not line:
            break
        
        line = line.lower()
        
        of.write(line)
    of.flush()
    of.close()

def getLineNum(fileName):

    filePtr = open(fileName, 'rb')
    count = 0

    while True:
        buffer = filePtr.read(81920 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    filePtr.close()


    return count
            
def listDirWithPattern(dirStr, listPattern = '.*'):

    if not os.path.exists(dirStr):
        print 'dir ' + dirStr + ' not exist'
        return []
 
    openList = []
    closeSet = {}
    pattern = re.compile(listPattern)
    
    ret = []
    closeSet[dirStr] = 1
    openList.append(dirStr)

    while len(openList) > 0:
        tmpDir = openList.pop()
        midList = os.listdir(tmpDir)
        for item in midList:
            fullPath = os.path.join(tmpDir, item)
            if os.path.isdir(fullPath):
                if not item in closeSet:
                    openList.append(fullPath)
                    closeSet[fullPath] = 1
            else:
                match = pattern.match(fullPath)
                if match:
                    ret.append(fullPath)
    
    return ret

if __name__ == '__main__':
    file = '/home/zhangjl/aiGit/asr/src/mxnet/train2.csv'
    lineNum = getLineNum(file)
    print 'line num is %d' %(lineNum)
