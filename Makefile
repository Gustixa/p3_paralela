NVCC = nvcc
CC = g++
CFLAGS = -std=c++11

all: hough

hough: houghBase.o cpu_hough.o pgm.o utils.o
	$(NVCC) -o hough houghBase.o cpu_hough.o pgm.o utils.o

houghBase.o: houghBase.cu
	$(NVCC) -c houghBase.cu

cpu_hough.o: cpu_hough.cpp cpu_hough.h
	$(CC) $(CFLAGS) -c cpu_hough.cpp

pgm.o: pgm.cpp pgm.h
	$(CC) $(CFLAGS) -c pgm.cpp

utils.o: utils.cpp utils.h
	$(NVCC) -c utils.cpp  # Cambia g++ por nvcc aquí

clean:
	rm -f *.o hough
