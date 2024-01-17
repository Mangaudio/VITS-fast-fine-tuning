#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rsync -avz --progress -e "ssh -p $2" --exclude-from=rsync_exclude.txt $SCRIPT_DIR/ root@$1:/root/VITS-fast-fine-tuning