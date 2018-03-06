NVCC = nvcc

TARGETS = \
pageable/main \
pageable_vs_pinned/main \
pinned/main

all: $(TARGETS) 

%: %.cu
	$(NVCC) $^ -std=c++11 -G -g -o $@ -ldl -lcuda -lnvToolsExt

clean:
	rm -f *.o $(TARGETS)
