/**
* @file window.hpp
* @brief Clase Window para el procesamiento de la Transformada de Hough en CPU y GPU.
*/
#include "window.hpp"

// Cuda Kernel
extern "C" void GPU_HoughTran(float* pcCos, float* pcSin, unsigned char* pic, int width, int height, int* acc, float rMax, float rScale);

const int   DEG_INCREMENT = 2;
const int   DEG_BINS =  static_cast<int>(180.0 / static_cast<double>(DEG_INCREMENT));
const int   RAD_BINS = 100;
const float RAD_INCREMENT = static_cast<float>(DEG_INCREMENT * M_PI / 180.0);

/**
 * @brief Constructor de la clase Window.
 *
 * Configura la interfaz gráfica, inicializa imágenes y realiza la primera ejecución de la Transformada de Hough en la CPU.
 */
Window::Window() {
	image_a = new QLabel(this);
	image_b = new QLabel(this);
	image_c = new QLabel(this);
	image_d = new QLabel(this);

	THRESHOLD = 600;

	input_image = Image("./runway.png");

	QSlider* slider = new QSlider(Qt::Horizontal, this);
	slider->setRange(0, 1200);
	slider->setValue(600);
	
	QLabel* threshold = new QLabel("600");

	QWidget* grid_widget = new QWidget(this);
	QGridLayout* grid_layout = new QGridLayout(grid_widget);

	QWidget* widget = new QWidget(this);
	QVBoxLayout* layout = new QVBoxLayout(widget);

	QHBoxLayout* header = new QHBoxLayout();

	header->addWidget(slider);
	header->addWidget(threshold);

	layout->addLayout(header);
	layout->addWidget(grid_widget);

	grid_layout->addWidget(image_a, 0, 0);
	grid_layout->addWidget(image_b, 0, 1);
	grid_layout->addWidget(image_c, 1, 0);
	grid_layout->addWidget(image_d, 1, 1);

	setCentralWidget(widget);

	cpu_output_image_a = Image::paramCopy(input_image);
	cpu_output_image_b = Image::paramCopy(input_image);
	cpu_output_image_overlay_a = Image::copy(input_image);
	cpu_output_image_overlay_b = Image::copy(input_image);

	const int width = input_image.width;
	const int height = input_image.height;

	auto start = std::chrono::high_resolution_clock::now();
	CPU_HoughTran(input_image, cpu_output_image_a, cpu_output_image_b, cpu_output_image_overlay_a, cpu_output_image_overlay_b, THRESHOLD, &cpuht);
	printf("CPU delta: %f", std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count());

	image_a->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_a.pixels.data(), width, height, width, QImage::Format_Grayscale8)));
	image_b->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_b.pixels.data(), width, height, width, QImage::Format_Grayscale8)));
	image_c->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_overlay_a.pixels.data(), width, height, width, QImage::Format_Grayscale8)));
	image_d->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_overlay_b.pixels.data(), width, height, width, QImage::Format_Grayscale8)));

	precomputeTrig(static_cast<int>(DEG_BINS), RAD_INCREMENT, &pcCos, &pcSin);
	QObject::connect(slider, &QSlider::valueChanged, [this, slider, threshold]() {
		THRESHOLD = slider->value();
		threshold->setText(QString::number(THRESHOLD));

		const int width = input_image.width;
		const int height = input_image.height;
		cpu_output_image_a = Image::paramCopy(input_image);
		cpu_output_image_b = Image::paramCopy(input_image);
		cpu_output_image_overlay_a = Image::copy(input_image);
		cpu_output_image_overlay_b = Image::copy(input_image);

		CPU_HoughTran(input_image, cpu_output_image_a, cpu_output_image_b, cpu_output_image_overlay_a, cpu_output_image_overlay_b, THRESHOLD, &cpuht);

		image_a->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_a.pixels.data(), width, height, width, QImage::Format_Grayscale8)));
		image_b->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_b.pixels.data(), width, height, width, QImage::Format_Grayscale8)));
		image_c->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_overlay_a.pixels.data(), width, height, width, QImage::Format_Grayscale8)));
		image_d->setPixmap(QPixmap::fromImage(QImage(cpu_output_image_overlay_b.pixels.data(), width, height, width, QImage::Format_Grayscale8)));
	});
}

/**
 * @brief Destructor de la clase Window.
 *
 * Libera la memoria asignada en GPU y CPU, y realiza el procesamiento final en CUDA.
 */
Window::~Window() {
	processCuda();
	cudaFree(d_in);
	cudaFree(d_hough);
	free(pcCos);
	free(pcSin);
	free(gpuht);
	free(cpuht);
}

/**
 * @brief Ejecuta la Transformada de Hough en GPU.
 *
 * Configura la memoria en GPU, copia los datos de la imagen y ejecuta el kernel de CUDA para calcular
 * la Transformada de Hough. Luego, compara los resultados de CPU y GPU.
 */
void Window::processCuda() {
	const int width = input_image.width;
	const int height = input_image.height;

	const float rMax = (float)(sqrt(1.0 * width * width + 1.0 * height * height) / 2.0);
	const float rScale = 2.0f * rMax / RAD_BINS;

	// Finish With Cuda
	cudaMalloc((void **) &d_in, sizeof(unsigned char) * width * height);
	cudaMalloc((void **) &d_hough, sizeof(int) * DEG_BINS * RAD_BINS);
	cudaMemcpy(d_in, input_image.pixels.data(), sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);
	cudaMemset(d_hough, 0, sizeof(int) * DEG_BINS * RAD_BINS);

	const int blockNum = static_cast<int>(ceil(width * height / 256.0));

	auto start = std::chrono::high_resolution_clock::now();
	GPU_HoughTran(pcCos, pcSin, d_in, width, height, d_hough, rMax, rScale);

	gpuht = (int*) malloc(DEG_BINS * RAD_BINS * sizeof(int));
	cudaMemcpy(gpuht, d_hough, sizeof(int) * DEG_BINS * RAD_BINS, cudaMemcpyDeviceToHost);

	printf("\nGPU delta: %f", std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count());

	for (int i = 0; i < DEG_BINS * RAD_BINS; i++) {
		if (cpuht[i] != gpuht[i]) {
			//printf("Mismatch at [%i] %i != %i\n", i, cpuht[i], gpuht[i]);
		}
	}

	//cpu_output_image_a.saveImage("./output_a.png");
	//cpu_output_image_b.saveImage("./output_b.png");
	//cpu_output_image_overlay_a.saveImage("./output_overlay_a.png");
	//cpu_output_image_overlay_b.saveImage("./output_overlay_b.png");

	printf("\nFinished");
}