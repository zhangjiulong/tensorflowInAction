#coding=utf-8

import math
import wave
import sys

def calMels(freq):
    ret = 1125.0 * math.log(1.0 + freq / 700.0, math.e)
    return ret

def calHz(mel):
    ret = 700.0 * (math.exp(mel / 1125.0) - 1.0)
    return ret

def calBin(hz):
    ret = math.floor((256.0 + 1.0) * hz/ 8000) # need to check 256 why not 512 
    return ret

def getWavLen(wavName):
    try:
        f = wave.open(wavName, "rb")
        
        params = f.getparams()
        frameRate = params[2]
        nFrames = params[3]
        wavLen = (nFrames * 1.0) / (frameRate * 1.0)
    except:
        info=sys.exc_info() 
        print info
        return -1.0
    return wavLen

if __name__ == '__main__':
    wavFile = '/home/zhangjl/dataCenter/asr/td/vx/vx/123413330.wav'
    print 'file len is ' + str(getWavLen(wavFile))
