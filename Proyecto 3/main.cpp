#include "window.hpp"

int main(int argc, char **argv) {
	QApplication* app = new QApplication(argc, argv);
	Window* window = new Window();

	window->showMaximized();
	app->exec();
	
	delete window;
	delete app;
	return 0;
}