#coding=utf-8
from fileTool import *
import codecs

def splitStmTxt(line):
    if len(line.split(' ')) <= 6:
        return False, '', ''
    retPre = ''
    retPost = ''
    splits = line.split(' ', 6)
    retPre = ' '.join(splits[:6])
    retPost = splits[6]
    return True, retPre, retPost

def staticalTotalTime(inStmDir):
    inStmList = listDirWithPattern(inStmDir, '.*stm$')
    fileReadNum = 0
    total = 0.0
    
    for item in inStmList:
        fileReadNum = fileReadNum + 1
        if fileReadNum % 100 == 0:
            print ' %d files has been processed.' %(fileReadNum)
            
        f = codecs.open(item, 'r', 'utf-8')
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue
                
            splits = line.split()
            startTime = float(splits[3])
            endTime = float(splits[4])
            used = endTime - startTime
            total = total + used

        f.close()
    return total




if __name__ == '__main__':
    #line = '1157599 A 1157599 35.5 39.499 <o,f0,female> 呃是公众号<sil> 呃就公众平台'
    #status, pre, post = splitStmTxt(line)
    ret = staticalTotalTime('/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stm/stmSeg5')
    print 'total time len is %f' %(ret/3600.0)

