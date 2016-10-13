#!/bin/bash

export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cudnn-4.0/cuda/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1 python3.5 translate1.py
