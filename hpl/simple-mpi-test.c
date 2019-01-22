#include    /* PROVIDES THE BASIC MPI DEFINITION AND TYPES */
#include

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv); /*START MPI */

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello world from MPI RANK %d\n", rank);

  MPI_Finalize();  /* EXIT MPI */
}
