# Configuración para CUDA y compilación con OpenCV
NVCC = nvcc
CC = g++
CFLAGS = -std=c++11

all: hough

hough: houghBase.o cpu_hough.o image.o utils.o
	$(NVCC) -o hough houghBase.o cpu_hough.o image.o utils.o

houghBase.o: houghBase.cu
	$(NVCC) -c houghBase.cu

cpu_hough.o: cpu_hough.cpp cpu_hough.h
	$(CC) $(CFLAGS) -c cpu_hough.cpp

image.o: image.cpp image.h stb_image.h  # Cambiado de pgm.o
	$(CC) $(CFLAGS) -c image.cpp

utils.o: utils.cpp utils.h
	$(NVCC) -c utils.cpp

clean:
	rm -f *.o hough
