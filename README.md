# reverse-engineer

## Valgrind / Callgrind

    valgrind --tool=callgrind
    kcachegrind

Can use `kcachegrind` to see what calls each CUDA API makes.

## NVIDIA Profiler

    `nvprof -o timeline.nvvp -f <exe>` - find API calls that are misbehaving

## Radare2

* aaa
* aaaa
* `afl` - list functions
* `sf` - step to next function

## Building Examples

    make