#encoding=utf-8
import os
import codecs
from fileTool import *
from wavTool import *


from BaseFormatConvertor import BaseFormatConvertor

class Tsing2TedliumFormat(BaseFormatConvertor):
    def __init__(self, srcDir, dstDir):
        self.__srcDir = srcDir
        self.__dstDir = dstDir

    def checkTxtFormat(self, txtFile):

        f = codecs.open(txtFile, 'r')
        ret = True
        lineNum = 0

        for line in f:
            lineNum = lineNum + 1
        
        if lineNum != 3:
            ret = False
        else:
            ret = True
        
        f.close()

        return ret

        
    def checkDirTxtFormat(self):
        txtList = listDirWithPattern(self.__srcDir, '.*trn')
        fileReadNum = 0

        for item in txtList:
            fileReadNum = fileReadNum + 1
            if fileReadNum % 100 == 0:
                print 'processing line %d' %(fileReadNum)

            checkResult = self.checkTxtFormat(item)
            
            if not checkResult:
                print 'file %s format is not ok' %(item)
    
    def readTxtFromTrn(self, trnFileName):
        ret = ''
        f = codecs.open(trnFileName, 'r', 'utf-8')
        txt = f.readline()
        f.close()
        txt = txt.replace(' ','')
        ret = txt.strip()
        return ret

    def convertTrn2Stm(self, trnFileName, wavFileName, stmFileName):
        wavPureFileNameNoExt = getPureFileNameWithOutExt(wavFileName)

        txtLine = self.readTxtFromTrn(trnFileName)
        startTime = 0.0
        endTime = getWavLen(wavFileName)
        if endTime < 0:
            print 'file %s len is too small.' %(wavFileName)
            return
        
        line2write = '%s A %s %f %f <o,f0,unknown> %s'%(wavPureFileNameNoExt, wavPureFileNameNoExt, startTime, endTime, txtLine)
        stmFile = codecs.open(stmFileName, 'w', 'utf-8')
        stmFile.write(line2write)
        stmFile.close()

    def convert(self):
        wavList = listDirWithPattern(self.__srcDir, '.*wav$')
        for item in wavList:
            wavFile = item
            trnFile = wavFile + '.trn'
            stmFile = re.sub(r'wav$', 'stm', wavFile)
            self.convertTrn2Stm(trnFile, wavFile, stmFile)
    
    def rmUnderlineFromFileName(self, srcDir, dstDir):
        fileList = listDirWithPattern(srcDir, '.*')
        for item in fileList:
            fileName = os.path.split(item)[1]
            fileName = fileName.replace('_', '')
            os.rename(item, os.path.join(dstDir, fileName))
                      
if __name__ == '__main__':
                      
    srcDir = '/asrDataCenter/dataCenter/asr/thchs-30/data_thchs30/data_renamed'
    dstDir = '/asrDataCenter/dataCenter/asr/thchs-30/data_thchs30/dataTedFormat'

    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    tsing2TedliumFormat = Tsing2TedliumFormat(srcDir,dstDir)
    tsing2TedliumFormat.convert()
    #tsing2TedliumFormat.rmUnderlineFromFileName('/asrDataCenter/dataCenter/asr/thchs-30/data_thchs30/data_renamed', '/asrDataCenter/dataCenter/asr/thchs-30/data_thchs30/data_renamed')
                      
    
    #tsing2TedliumFormat.checkDirTxtFormat()
