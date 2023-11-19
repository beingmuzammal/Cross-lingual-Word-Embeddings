CC = gcc
#For older gcc, use -O3 or -O2 instead of -Ofast
CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result

all: $(patsubst %.c, %.out, $(wildcard *.c))
%.out: %.c makefile
	$(CC) $< -o $@ $(CFLAGS)
clean:
	rm *.out