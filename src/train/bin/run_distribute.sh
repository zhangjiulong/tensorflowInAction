#!/bin/bash

if [ $# != 1 ];then
   echo $0" run"
   echo $0" kill"
   echo $0" status"
   echo $0" sendfile"
   exit 0
fi

deploy_dir="/home/luodongri/code/git/asr/src/train/bin"
deploy_dir2="/home/luodongri/code/git/asr/src/train/bin2"
deploy_dir3="/home/luodongri/code/git/asr/src/train/bin3"

local_data_dir="/data/200h/"
data_dir="/data/200h/"
data_dir2="/data/200h2/"
data_dir3="/data/200h3/"

ps_host="192.168.100.41:2200"

#格式是ip:port:第几个gpu
worker_host="192.168.100.41:2210,192.168.100.41:2211,192.168.100.41:2212,192.168.100.42:2220"

host_info="--ps_hosts="$ps_host" --worker_hosts="$worker_host


#ssh到每台机器上， kill到进程
if [ "$1" == "kill" ];then
   array_str=${worker_host//,/ }
   for host in $array_str; do
    host_array=(${host//:/ })
    ip=${host_array[0]}
    ssh $ip "ps aux | grep python | grep asr_distribute.py | awk '{print \$2;}' | xargs kill -9 "
	sleep 1s
    done
	exit 0
fi


# ssh到每台机器上， 查看有几个进程还在运行，不过第一个机器正常情况下会多2个, 一个是ps进程，一个是执行命令本身造成的
if [ "$1" == "status" ];then
   array_str=${worker_host//,/ }
   for host in $array_str; do
    host_array=(${host//:/ })
    ip=${host_array[0]}
  wcl=$(ssh $ip "ps aux | grep python | grep asr_distribute.py "  | wc -l)
  processnum=$(expr $wcl - 1)
  echo $ip" "$processnum
  sleep 1s
    done
  exit 0
fi

if [ "$1" == "sendfile" ];then
  #自己写命令到这里
  scp $local_data_dir"/train.eesen.shuffle.3" 192.168.100.42:$data_dir
  scp $local_data_dir"/train.eesen.shuffle.4" 192.168.100.42:$data_dir
  scp $local_data_dir"/train.eesen.shuffle.5" 192.168.100.42:$data_dir2
  scp $local_data_dir"/train.eesen.shuffle.6" 192.168.100.42:$data_dir2
  
  scp $local_data_dir"/train.eesen.shuffle.7" 192.168.100.22:$data_dir
  scp $local_data_dir"/train.eesen.shuffle.8" 192.168.100.62:$data_dir
  scp $local_data_dir"/train.eesen.shuffle.9" 192.168.100.62:$data_dir2

fi



#开始执行命令
if [ "$1" == "run" ];then
#先ssh到ps服务器， 执行ps命令
echo "execute ps server command"
ssh 192.168.100.41 "cd "${deploy_dir}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices='' "$host_info" --job_name=ps --task_index=0 >nohup 2>&1 &"
sleep 1s
echo "execute ps server command over"

#执行worker节点
echo "execute worker server command"

echo 1
ssh 192.168.100.41 "cd "${deploy_dir}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices=0 "$host_info" --job_name=worker --data_dir=/data/200h/ --train_file_number=6 --task_index=0 >nohup2 2>&1 &"
sleep 1s

echo 2
ssh 192.168.100.41 "cd "${deploy_dir2}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices=1 "$host_info" --job_name=worker --data_dir=/data/200h2/ --train_file_number=6  --task_index=1 >nohup 2>&1 &"
sleep 1s

echo 3
ssh 192.168.100.41 "cd "${deploy_dir3}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices=2 "$host_info" --job_name=worker --data_dir=/data/200h3/ --train_file_number=6  --task_index=2 >nohup 2>&1 &"
sleep 1s

echo 4
ssh 192.168.100.42 "cd "${deploy_dir}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices=0 "$host_info" --job_name=worker  --data_dir=/data/200h/ --train_file_number=6 --task_index=3 >nohup 2>&1 &"
sleep 1s

echo "execute worker server command over"
fi
