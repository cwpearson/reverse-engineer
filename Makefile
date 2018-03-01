NVCC = nvcc

all: main

main: src/main.cu
	$(NVCC) $^ -std=c++11 -G -g -o $@

clean:
	rm -f *.o main