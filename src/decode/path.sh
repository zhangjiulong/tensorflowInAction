# 这个脚本的目的就是导入各种路径, 让解码工作中的各个命令可以调用.

export EESEN_ROOT=/home/rukuang/software/eesen
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/irstlm/bin/:$PWD:$PATH
export LC_ALL=C
