#! /bin/bash

set -eu

MACHINE=$1

make

for b in pageable_vs_pinned; do
    mkdir -pv "$b"/results/"$MACHINE"
    valgrind --tool=callgrind \
        --log-file="$b"/results/"$MACHINE"/callgrind.log \
        --callgrind-out-file="$b"/results/"$MACHINE"/callgrind.out \
        "$b"/main
done