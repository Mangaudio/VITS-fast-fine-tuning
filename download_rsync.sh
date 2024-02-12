#!/bin/bash

# 远程主机IP地址
remote_host=$1
# 远程主机SSH端口号
ssh_port=$2
# 本地保存路径
local_path="./remote_result/"

# 在远程主机执行命令 head /root/VITS-fast-fine-tuning/short_character_anno.txt -n 1 | awk -F'|' '{print $2}'， 将结果保存到变量 speaker_name 中
speaker_name=$(ssh -p $ssh_port root@$remote_host "head /root/VITS-fast-fine-tuning/short_character_anno.txt -n 1 | awk -F'|' '{print \$2}'")

# 从OUTPUT_MODEL/中找出最大值
epochs=$(( $(ssh -p $ssh_port root@$remote_host "ls /root/VITS-fast-fine-tuning/OUTPUT_MODEL/G_*.pth | grep -v 'G_latest.pth' | sed 's/.*G_\([0-9]*\).pth/\1/' | sort -nr | head -n 1") + 100 ))

echo Speaker name: $speaker_name
echo Epochs: $epochs

# 创建本地保存路径
mkdir -p $local_path

# 下载 G_latest.pth 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/OUTPUT_MODEL/G_latest.pth $local_path/G_latest_${speaker_name}_$epochs.pth

# 下载 D_latest.pth 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/OUTPUT_MODEL/D_latest.pth $local_path/D_latest_${speaker_name}_$epochs.pth

# 下载 finetune_speaker.json 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/configs/modified_finetune_speaker.json $local_path/modified_finetune_speaker_${speaker_name}_$epochs.json

# 下载 short_character_anno.json 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/short_character_anno.txt $local_path/short_character_anno_${speaker_name}_$epochs.txt