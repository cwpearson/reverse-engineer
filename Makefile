NVCC = nvcc

all: main

main: pageable_vs_pinned/main.cu
	$(NVCC) $^ -std=c++11 -G -g -o $@ -ldl -lcuda

clean:
	rm -f *.o main