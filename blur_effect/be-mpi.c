// mpicc mpi-omp_pi.c -o mpi-omp_pi -lm -fopenmp
// mpirun -np 4 --hostfile mpi-hosts ./mpi-omp_pi
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
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

 
#define ITERATIONS 2e05
#define MAXTHREADS 32
 
int calculatePi(double *pi, int numprocs, int processId)
{   int start, end;
    printf("processId: %d \n",processId);
    start = (long)(ITERATIONS/numprocs)*processId;
    printf( "Start is : %d \n" ,start);
    end = (long)(ITERATIONS/numprocs) * (1+processId);
    printf( "End is : %d \n" ,end);
    int i = start;
 
    do{
        *pi = *pi + (double)(4.0 / ((i*2)+1));
        i++;
        *pi = *pi - (double)(4.0 / ((i*2)+1));
        i++;
    }while(i < end);
    printf("Pi local: %.10f\n",*pi); 
    return 0;
}
 
 
 
int main(int argc, char *argv[])
{
    int done = 0, n, processId, numprocs, I, rc, i;
    double PI25DT = 3.141592653589793238462643;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    if (processId == 0) printf("\nLaunching with %i processes", numprocs);
    double local_pi[numprocs], global_pi;
    global_pi = 0.0;
    printf("%d \n", processId);
    calculatePi(&local_pi[processId], numprocs, processId);
   printf("Local pi es: %.16f\n", &local_pi[processId]);
  MPI_Reduce(local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (processId == 0) printf("\npi is approximately %.16f, Error is %.16f\n", global_pi, fabs(global_pi - PI25DT));
    MPI_Finalize();
    return 0;
}
