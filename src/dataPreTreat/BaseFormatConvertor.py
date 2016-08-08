#coding=utf-8

class BaseFormatConvertor():
    def __init__(self, srcDir, dstDir):
        self.__srcDir = srcDir
        self.__dstDir = dstDir

    @property
    def srcDir(self):
        return self.__srcDir

    @srcDir.setter
    def srcDir(self, v):
        self.__srcDir = v

    @property
    def dstDir(self):
        return self.__dstDir
        
    @dstDir.setter
    def dstDir(self, v):
        self.__dstDir = v

    
    def convert(self):
        print 'src is ' + self.__srcDir

