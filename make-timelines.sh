#! /bin/bash

set -eu

MACHINE=$1

make

for b in pageable_vs_pinned pageable pinned; do
    mkdir -pv "$b"/results/"$MACHINE"
    nvprof -o "$b"/results/"$MACHINE"/timeline.nvvp -f "$b"/main
done