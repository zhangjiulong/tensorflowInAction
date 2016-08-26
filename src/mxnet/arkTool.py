#coding=utf-8
import codecs
import re
import logging

def arkTxt2Csv(inFile, outFile):
    inFilePtr = codecs.open(inFile, 'r', 'utf-8')
    outFilePtr = codecs.open(outFile, 'w', 'utf-8')

    delPattern = re.compile(r'^.*\[')
    writePattern = re.compile(r'.*\]')
    
    writeStr = ''
    lineNum = 0
    for line in inFilePtr:
        lineNum = lineNum + 1
        line = line.strip()
        
        # 1. del empty line
        if len(line) <= 0: # del if line is empty
            logging.error('empty line on line ' + str(lineNum))
            continue
        
        # 2. del head line
        match = delPattern.match(line)
        if match:
            logging.info('head line on line ' + str(lineNum))
            continue

        line = line.replace('\n', '')
        writeStr = writeStr + ' ' + line
        writeMatch = writePattern.match(line)
        
        if writeMatch:
            writeStr = writeStr.replace(']', '')
            writeStr = writeStr.strip()
            writeStr = writeStr.replace(' ', ',')
            outFilePtr.write(writeStr + '\n')
            writeStr = ''

    inFilePtr.close()
    outFilePtr.close()

    
def labelTxt2Csv(inFile, outFile):
    inFilePtr = codecs.open(inFile, 'r', 'utf-8')
    outFilePtr = codecs.open(outFile, 'w', 'utf-8')
    #labelPattern = '[^ \-]+\-[^ \-]+
    lineNum = 0
    
    for line in inFilePtr:
        lineNum = lineNum + 1
        line = line.strip()
        
        splits = line.split(' ', 1)
        labelStr = splits[1].strip()
        labelStr = labelStr.replace(' ', ',')
        outFilePtr.write(labelStr + '\n')
        
        
if __name__ == '__main__':
    arkFile = './ark.csv'
    arkOutFile = './ark.out.csv'
    arkTxt2Csv(arkFile, arkOutFile)
