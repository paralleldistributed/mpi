#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct uchar3 {
	uchar x;
	uchar y;
	uchar z;
};
struct float3 {
	float x;
	float y;
	float z;
};
struct args {
	uchar3 *src, *ans;
	int cols, rows, n_threads, id_thread, radio;
};


uchar3 *ans;
const uchar3 *origin;

/**
* Convierte i a una cordenada de la forma (x,y).
* Retorna un apuntador con 2 pociciones reservadas.
* En la primera almacena el valor de x
* En la segunda almacena el valor de y
*/
int *iToxy(int i, int cols) {
	int *ans;
	ans = (int*)malloc(2 * sizeof(int));
	*ans = i%cols;
	*(ans + 1) = i / cols;
	return ans;
}

/**
* convierte una cordenada (x,y) a un valor i para array
* Retorna un entero con el valor de i
*/
int xyToi(int x, int y, int cols) {
	return cols*y + x;
}

/**
* Halla la suma promediada de los pixeles vecinos en base a un kernel
* src*			Un apuntador a el vector de datos de la imagen
* pos:			El indice del pixel, el indice en base a un array unidimencional
* rows, cols:	dimenciones de la imagen que se esta procesando
* radio:		El radio del kernel para los pixeles vecinos
* Retorna un entero con el valor de i
*/
uchar3 prom_punto(int pos, int rows, int cols, int radio) {
	float  sum_peso;
	float3 sum = { 0,0,0 };

	sum_peso = 0;

	int *ptr_aux = iToxy(pos, cols);
	int x = *ptr_aux;
	int y = *(ptr_aux + 1);
	free(ptr_aux);

	for (int k = -radio; k <= radio; k++) {
		for (int j = -radio; j <= radio; j++) {
			if ((x + k) >= 0 && (x + k) < cols &&
				(y + j) >= 0 && (y + j) < rows) {
				float peso = exp(-(k*k + j*j) / (float)(2 * radio*radio)) / (3.141592 * 2 * radio*radio);

				sum.x += peso * (origin + xyToi(x + k, y + j, cols))->x;
				sum.y += peso * (origin + xyToi(x + k, y + j, cols))->y;
				sum.z += peso * (origin + xyToi(x + k, y + j, cols))->z;
				sum_peso += peso;
			}
		}
	}

	uchar3 ans;

	ans.x = (uchar)std::floor(sum.x / sum_peso);
	ans.y = (uchar)std::floor(sum.y / sum_peso);
	ans.z = (uchar)std::floor(sum.z / sum_peso);

	return ans;
}

/**
* Recorre los puntos del vector de datos de la imagen haciendo el blur a cada uno de ellos
*/
void thread_Blur(const args *arg) {
	for (int i = arg->id_thread; i < arg->cols*arg->rows; i += arg->n_threads) {
		uchar3 aux = prom_punto(i, arg->rows, arg->cols, arg->radio);
		(ans + i)->x = aux.x;
		(ans + i)->y = aux.y;
		(ans + i)->z = aux.z;
	}
	return;
}

int main(int n, char* argv[])
{
	int THREADS = 0;
	int img_size = 0;
	float radio = -1;
	cv::Mat src;

	//	Prueba que los parametros esten completos
	if (n != 4) {
		printf("blur <ruta img> <kernel> <thread>\n");
		return 0;
	}

	//	Determina el radio del kernel
	std::stringstream ss;
	ss << argv[2];
	ss >> radio;
	if (radio < 1) {
		std::cerr << " Radio incorrecto para el Kernel, debe ser mayor a 1\n";
		return -1;
	}
	std::cout << "Kernel radio: " << (int)floor(radio) << std::endl;

	//Cuántos hilos se usaran
	ss.clear();
	ss << argv[3];
	ss >> THREADS;
	if (THREADS < 1) {
		std::cerr << " Numero de hilos no permitido, debe ser mayor a 1\n";
		return -1;
	}
	std::cout << "Hilos: " << (int)floor(THREADS) << std::endl;

	//	Carga la imagen en memoria
	src = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!src.data) {
		std::cerr << "Error al leer la imagen\n";
		return -1;
	}
	origin = (uchar3*)src.data;
	std::cout << "Imagen: " << src.cols << "x" << src.rows << std::endl;

	//	Determina el tamaño del bloque de memoria para la imagen
	img_size = src.cols*src.rows * sizeof(uchar3);
	std::cout << "Imagen: " << ((double)img_size) / 1e6 << " Mb." << std::endl;

	//	Reservar la memoria para imagen de respuesta
	ans = (uchar3*)malloc(img_size);
	if (ans == NULL) {
		std::cerr << "Error al reservar memoria para imagen ans en GPU\n";
		return -1;
	}
	std::cout << "Memoria de imagen ans \n";

	//	llamar procesos de blur paralelos
	std::vector<std::thread> threads;

	std::cout << "iniciando" << std::endl;
	for (int i = 0; i < THREADS; i++) {
		args *arg = new args;
		arg->src = (uchar3*)src.data;
		arg->ans = ans;
		arg->cols = src.cols;
		arg->rows = src.rows;
		arg->n_threads = THREADS;
		arg->radio = radio;
		arg->id_thread = i;
		threads.push_back(std::thread(thread_Blur, arg));
	}
	for (std::thread &t : threads) {
		t.join();
	}

	std::cout << "Fin threads\n";
	//Liberando y reasignando memoria
	//free(src.data);
	src.data = (uchar*)ans;
	std::cout << "Memoria liberada\n";

	//	namedWindow("final");
	//	imshow("final", dst);

	/*cv::namedWindow("initial");*/
	/*cv::imshow("initial", src);*/
	imwrite("./thread_blur.jpg", src);

	/*cv::waitKey(0);*/

	return 0;
}