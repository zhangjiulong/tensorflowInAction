#!/usr/bin/env python


a=100000
#print(`a`)
#print(repr(100000L))

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
data_set = [[] for _ in _buckets]
print("%s"%data_set)

for bucket_id, (source_size, target_size) in enumerate(_buckets):
    print("%d  %d  %d" % (bucket_id,source_size,target_size))


a=['aa',1]
b=['bb',2]
c=[a,b]
print(c)

a=[1,2,3,4,5,6,7,8,9]
print(a[:-1])

def test(dd):
  print(dd)

test([2,3])





a="年 发 顺 丰 的"
for i in a.split():
  print(i)
a="ab llcd edf"
for i in a.split():
  print(i)
