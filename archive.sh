#!/bin/bash
cmd=$1
# cmd is always required
if [ -z "$cmd" ] ; then
    echo "Usage: $0 <cmd> [<name>]"
    exit 1
fi

name=$2

# name is required for save and load
if [ "$cmd" == "save" ] || [ "$cmd" == "load" ]; then
    if [ -z "$name" ]; then
        echo "Usage: $0 <cmd> [<name>]"
        echo "name is required for save and load"
        exit 1
    fi
fi

# command includes save, list, load, clean
# save
if [ "$cmd" == "save" ]; then
    target=./characters/$name
    # if target exists, ask for overwrite
    if [ -d "$target" ]; then
        read -p "character $name already exists, overwrite? [y/n] " ans
        if [ "$ans" != "y" ]; then
            exit 1
        fi
    fi
    mkdir -p $target

    cp -r OUTPUT_MODEL $target/OUTPUT_MODEL
    cp -r configs/modified_finetune_speaker.json $target/finetune_speaker.json
    cp -r final_annotation_* $target/
    cp -r short_character_anno.txt $target/
elif [ "$cmd" == "load" ]; then
    target=./characters/$name
    if [ ! -d "$target" ]; then
        echo "character $name not found"
        exit 1
    fi

    cp -r $target/OUTPUT_MODEL OUTPUT_MODEL
    cp -r $target/finetune_speaker.json configs/modified_finetune_speaker.json
    cp -r $target/final_annotation_* .
    cp -r $target/short_character_anno.txt .
elif [ "$cmd" == "list" ]; then
    ls ./characters
elif [ "$cmd" == "clean" ]; then
    rm -r OUTPUT_MODEL configs/modified_finetune_speaker.json final_annotation_* short_character_anno.txt
else
    echo "Usage: $0 <cmd> [<name>]"
    echo "cmd can be save, list, load, clean"
    exit 1
fi
