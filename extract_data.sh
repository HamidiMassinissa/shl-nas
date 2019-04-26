#!/bin/bash

for d in $(ls -d ./data/*)
do
    tar -xvf \
        $d \
        -C ./generated/tmp \
        --wildcards \
        "SHLDataset_preview_v1/*/*/Label.txt" \
        "SHLDataset_preview_v1/*/*/*_Motion.txt" \
        -k # do not overwrite already extracted files
done
