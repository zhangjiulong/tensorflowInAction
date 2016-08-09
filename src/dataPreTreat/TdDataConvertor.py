#coding=utf-8
import os
import re
import shutil
import sys
import imp
#import jieba
import string
imp.reload(sys)
from fileTool import *
from BaseFormatConvertor import BaseFormatConvertor
from paAsrTools import * 
import codecs
#sys.setdefaultencoding('utf-8')

class Td2TedliumFormat(BaseFormatConvertor):

    def __init__(self, srcDir, dstDir):
        self.__srcDir = srcDir
        self.__dstDir = dstDir
        
    def displayError(self,fileStr, line):
        print 'file is ' + fileStr + ' line is ' + str(line) + ' error'

    
    # first check file head like -----\nA:  \nB:   \n --------
    def checkFileHeaders(self, fileName):
        f = codecs.open(fileName, 'r', 'utf-8')
        ret = []
        lineNum = 0

        # check first line
        firstPattern = re.compile(r'-----.*')
        line = f.readline()
        lineNum = lineNum + 1
        if not line:
            self.displayError(fileName, lineNum)
            return False

        line = line.strip()
            
        match = firstPattern.match(line)
        if not match:
            return False
            
        
        secPart = []

        # read last part of the head
        while 1:
            line = f.readline()
            line = line.strip()
            lineNum = lineNum + 1
            if not line:
                self.displayError(fileName, lineNum)
                return False
            match = firstPattern.match(line)
            if match:
                break
            
            secPart.append(line)

        f.close()
        # check validation
        if len(secPart) == 1:
            return True

        sum = 0
        for i in range(1,len(secPart)):
            tmp = ord(secPart[i][0]) - ord(secPart[i - 1][0])
            if tmp != 1 :
                self.displayError(fileName, lineNum)
                return False
            sum = sum + tmp
        
        tmpDif = ord(secPart[-1][0]) - ord(secPart[0][0])

        if sum != tmpDif:
            return False
        
        return True

    # check file body
    def checkBody(self, fileName):

        f = codecs.open(fileName, 'r', 'utf-8')
        ret = []
        lineNum = 0

        # check first line
        firstPattern = re.compile(r'-----.*')
        line = f.readline()
        lineNum = lineNum + 1
        if not line:
            self.displayError(fileName, lineNum)
            return False

        line = line.strip()
            
        match = firstPattern.match(line)
        if not match:
            return False
            
        
        secPart = []

        # read second -----
        while 1:
            line = f.readline()
            
            lineNum = lineNum + 1
            if not line:
                self.displayError(fileName, lineNum)
                return False
            line = line.strip()
            match = firstPattern.match(line)
            if match:
                break
        
        # begin check body
        while 1:
            line = f.readline()
            lineNum = lineNum + 1
            if not line:
                return True
            if line[0] != '[':
                return False
    
    def checkDir(self, dirStr):
        txtList = listDirWithPattern(dirStr, '.*/[0-9]+\.txt$')
        for item in txtList:
            headResult = self.checkFileHeaders(item)
            if not headResult:
                print 'in header %s' %(item)
            bodyResult = self.checkBody(item)
            if not bodyResult:
                print 'in body %s' %(item)

        
    def getPureFileNameWithOutExt(self, fileName):
        tmpLastPart = os.path.split(fileName)[1]
        ret = os.path.splitext(tmpLastPart)[0]
        return ret
        
    # extractStartTime(txtStr):
    def extractTimeAndTxt(self, txtStr):
        lastPattern = re.compile(r'\[([\d]+)[:;]([\d\.]+),([\d]+)[:;]([\d\.]+),?([A-Z]?)\](.*)$')
        match = lastPattern.match(txtStr)

        if match:
            startTime = float(match.group(1)) * 60.0 + float(match.group(2)) * 1.0
            endTime = float(match.group(3)) * 60.0 + float(match.group(4)) * 1.0
            sex = match.group(5)
            txt = match.group(6)
            return startTime, endTime, sex, txt
        else:
            return None,None,None,None,None
    
    # only process format is ok
    def convertTxt2Stm(self, fileName, stmFile):
        firstPattern = re.compile(r'-----.*')
        fn = self.getPureFileNameWithOutExt(fileName)
        wc = 'A'
        speakerID = fn
        #f = open(fileName)
        f = codecs.open(fileName, encoding='utf-8')
        stmFile = codecs.open(stmFile, 'w', 'utf-8')
        
        # read first '--------'
        f.readline()

        # read persons until '------'
        sexile = {}
        while 1:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            match = firstPattern.match(line)
            # if match then female and male are all read
            if match:
                break

            speakerSplits = line.split(':')
            if len(speakerSplits) != 2:
                print 'processing file ' + fileName + ' error'
                continue

            if speakerSplits[1][0] == 'f':
                sexile[speakerSplits[0]] = 'female'
            else:
                sexile[speakerSplits[0]] = 'male'

        # process lines like this '[0:06.650,0:08.057,A]你好很高兴为您服务'
        while 1:
            dic2rmduptime = {}
            line = f.readline()
            if not line:
                break
            line = line.strip()
            sex = 'U'
            try:
                startTime,endTime, sex, txt = self.extractTimeAndTxt(line)
            except:
                print 'filename is ' + fileName + ' line is ' + line + '\n'
                continue

            if not sex:
                #print 'file error ' + fileName
                sex = 'A'
            else:
                if sex in sexile:
                    pass
                else:
                    sex = 'A'
            if endTime <= startTime:
                print 'time error at file %s line %s '%(fileName, line)
                continue
            
            if endTime - startTime < 0.1:
                print 'short time error at file %s line %s '%(fileName, line)
                continue
            
            if dic2rmduptime.has_key(startTime):
                print 'dup time error at file %s line %s '%(fileName, line)
                continue

            dic2rmduptime[startTime] = 1
            line2write = ''
            try:
                line2writePre = fn + ' ' + wc + ' ' + speakerID + ' ' + str(startTime) + ' ' + str(endTime) + ' ' + '<o,f0,' + sexile[sex] + '>'
            except:
                print 'filename is ' + fileName + ' line is ' + line + '\n'
                continue
            
            # segment and remove noises
            #txt = re.sub('/[a-zA-Z]+', '[NOISE]', txt.strip())
            #txt = txt.lower()

            #segList = jieba.cut(txt)
            #txt = ' '.join(segList)
            #txt = txt.strip()
            #txt = txt.replace('[ NOISE ]', '[NOISE]')
            '''
            leftAlphaPattern = re.compile(r'.*[A-Za-z][ $].*')
            rightAlphaPattern = re.compile(r'^[a-z]')
            rightMatch = rightAlphaPattern.match(txt)
            leftMatch = leftAlphaPattern.match(txt)
            if rightMatch or leftMatch:
                print 'file is ' + fileName + ' line is alpha '  + txt
                continue
            '''
            #txt = txt.replace(string.punctuation, '')
            #txt = re.sub(r'[。？，]', '', txt)
            # after process txt write to file
            
            line2write = line2writePre + ' ' + txt + '\n'
            stmFile.write(line2write)
        
        stmFile.flush()
        stmFile.close()
            

    def convert(self):
        fileList = listDirWithPattern(self.__srcDir, '.*\.txt')
            
        for item in fileList:
            filename = os.path.split(item)[1]
            filename = filename.replace('.txt', '.stm')
            stmItem = os.path.join(self.__dstDir, filename)
            self.convertTxt2Stm(item, stmItem)

    def selectBothWavAndTxt(self, wavDir, txtDir):
        wavList = listDirWithPattern(wavDir, '.*\.wav')
        txtList = listDirWithPattern(txtDir, '.*\.txt')
        wavList = map(lambda x:os.path.splitext(os.path.split(x)[1])[0], wavList)
        txtList = map(lambda x:os.path.splitext(os.path.split(x)[1])[0], txtList)

        singleList = [x for x in wavList if x in txtList]
        for item in singleList:
            srcPath1 = os.path.join(wavDir, item + '.wav')
            shutil.copy(srcPath1, '/home/zhangjl/dataCenter/asr/td/vx/wav/')
            srcPath2 = os.path.join(txtDir, item + '.txt')
            shutil.copy(srcPath2, '/home/zhangjl/dataCenter/asr/td/vx/txt/')
            
    # collect special notions
    def collectSpecialNotions(self, stmDir):
        stmList = listDirWithPattern(stmDir, '.*stm$')
        notionDic = {}
        p = re.compile('.*(\/[a-zA-Z]+)', re.IGNORECASE)
        #p = re.compile('.*(\/n:[a-zA-Z\-]+)', re.IGNORECASE)
        fileReadNum = 0
        
        for item in stmList:
            fileReadNum = fileReadNum + 1
            if fileReadNum % 1000 == 0:
                print 'file processed %d' %(fileReadNum)
            f = codecs.open(item, 'r', 'utf-8')
            for line in f:
                line = line.strip()
                #match = p.match(line)
                #if match:
                tmpList = p.findall(line)
                for result in tmpList:
                    notionDic[result] = item
            f.close()
        for key in notionDic:
            print key

    def replaceLabels(self, inStmDir, outStmDir):
        inStmList = listDirWithPattern(inStmDir, '.*stm$')
        fileReadNum = 0

        for item in inStmList:
            fileReadNum = fileReadNum + 1
            if fileReadNum % 100 == 0:
                print ' %d files has been processed.' %(fileReadNum)
            
            f = codecs.open(item, 'r', 'utf-8')
            pureFileName = getPureFileName(item)
            fout = codecs.open(os.path.join(outStmDir, pureFileName), 'w', 'utf-8')
            for line in f:
                line = line.strip()
                line = line.replace('/noise', '/noise ')
                line = line.replace('/sil', '/sil ')
                line = line.replace('/mix', '/mix ')
                line = line.replace('/nps', '/nps ')
                line = line.replace('/non', '/non ')
                line = line.replace('/sil', '/sil ')

                line = line.replace('/n:lipsmack', '/n:lipsmack ')
                line = line.replace('/n:laughter', '/n:laughter ')
                line = line.replace('/n:cough', '/n:cough ')
                line = line.replace('/n:sneeze', '/n:sneeze ')
                line = line.replace('/n:throat-clear', '/n:throat-clear ')
                line = re.sub(r'[ ]+', ' ', line)
                line = line.strip()
                fout.write(line + '\n')

            f.close()
            fout.close()
        
if __name__ == '__main__':
    td2TedliumFormat = Td2TedliumFormat('/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/txt', '/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stm2')
    #td2TedliumFormat.checkDir('/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/txt')
    #td2TedliumFormat.checkBody('/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/txt/1500159.txt')
    
    #td2TedliumFormat.convert()
    #td2TedliumFormat.collectSpecialNotions('/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stm/stmNoSeg')
    #td2TedliumFormat.collectSpecialNotions('/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stm/stmLabelAlone2')
    td2TedliumFormat.replaceLabels("/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stm/stmNoSeg1", "/home/zhangjl/asrDataCenter/dataCenter/asr/td/vx/stm/stmLabelAlone2")
    #fileName = '/home/zhangjl/dataCenter/asr/td/vx/1/0924159.txt'
    #td2TedliumFormat.checkFileHeaders(fileName)

    #dirName = '/home/zhangjl/dataCenter/asr/td/vx/'
    #td2TedliumFormat.checkDir(dirName)
    #td2TedliumFormat.convertTxt2Stm('/home/zhangjl/dataCenter/asr/td/vx/txt_u8/1110498.txt', '/home/zhangjl/dataCenter/asr/td/vx/stm/1110498.stm')
    #td2TedliumFormat.convert()
    #str = '[0:40.928,0:42.043]/mix'
    #lastPattern = re.compile(r'\[([\d]+)[:;]([\d\.]+),([\d]+)[:;]([\d\.]+),?([A-Z]?)\](.*)$')

    
    #match = lastPattern.match(str)
    #txt = match.group(6)
    #print txt
    
    wavDir = '/home/zhangjl/dataCenter/asr/td/vx/wavOri/'
    txtDir = '/home/zhangjl/dataCenter/asr/td/vx/txtOri_u8/'
    #td2TedliumFormat.selectBothWavAndTxt(wavDir, txtDir)
