# Author: Kuang.Ru
# 主要用于统计STM对应音频文件的总时长.
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stm_dir", help="保存stm文件的文件夹路径.")
args = parser.parse_args()
seconds = 0

for root, dirs, file_names in os.walk(args.stm_dir):
    for filename in file_names:
        with open(os.path.join(root, filename), encoding="utf-8") as file:
            for line in file:
                start, end = [float(field) for field in line.split()[3:5]]
                seconds += end - start

print(seconds)
print(seconds / 3600, "小时")
