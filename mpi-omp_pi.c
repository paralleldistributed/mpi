//mpicc mpi-omp_pi.c -o mpi-omp_pi -lm -fopenmp
// mpirun -np 4 --hostfile mpi-hosts ./mpi-omp_pi
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
 
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
