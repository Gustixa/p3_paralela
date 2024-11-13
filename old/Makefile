# Configuración para CUDA y compilación con OpenCV
NVCC = nvcc
CC = g++
CFLAGS = -std=c++11
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`  # Cambia a opencv si usas una versión anterior a OpenCV 4
OPENCV_INCLUDE = -I/usr/include/opencv4  # Ajusta esta ruta si es diferente

all: hough

hough: houghBase.o cpu_hough.o image.o utils.o
	$(NVCC) -o hough houghBase.o cpu_hough.o image.o utils.o $(OPENCV_FLAGS)

houghBase.o: houghBase.cu
	$(NVCC) -c houghBase.cu $(OPENCV_INCLUDE)

cpu_hough.o: cpu_hough.cpp cpu_hough.h
	$(CC) $(CFLAGS) -c cpu_hough.cpp

image.o: image.cpp image.h stb_image.h
	$(CC) $(CFLAGS) -c image.cpp $(OPENCV_FLAGS)

utils.o: utils.cpp utils.h
	$(NVCC) -c utils.cpp

clean:
	rm -f *.o hough
