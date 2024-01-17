#!/bin/bash

# 远程主机IP地址
remote_host=$1
# 远程主机SSH端口号
ssh_port=$2
# 本地保存路径
local_path="./remote_result/"

# 创建本地保存路径
mkdir -p $local_path

# 下载 G_latest.pth 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/OUTPUT_MODEL/G_latest.pth $local_path/G_latest_$(date +%Y%m%d%H%M%S).pth

# 下载 D_latest.pth 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/OUTPUT_MODEL/D_latest.pth $local_path/D_latest_$(date +%Y%m%d%H%M%S).pth

# 下载 finetune_speaker.json 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/configs/modified_finetune_speaker.json $local_path/modified_finetune_speaker_$(date +%Y%m%d%H%M%S).json

# 下载 short_character_anno.json 文件并重命名
rsync -avz --progress -e "ssh -p $ssh_port" root@$remote_host:/root/VITS-fast-fine-tuning/short_character_anno.txt $local_path/short_character_anno_$(date +%Y%m%d%H%M%S).txt