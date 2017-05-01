#include <stdio.h>
#include <mpi.h>

#include <errno.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "tools/random.h"
#include "tools/test.h"

#define MAT_SIZE 1024
#define PROC_NUM 1024
#define PATH "trans.c"
#define get_shmid(x) shmget(ftok(PATH, x), MAT_SIZE * MAT_SIZE * sizeof(int), IPC_CREAT | 0666)

int main (int argc, char *argv[]) {
	/* Get rank for each process */
	int size, rank;
	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	/* Check if process number is PROC_NUM */
	if (size != PROC_NUM) {
		if (rank == 0)
			printf ("Run this program with option '(mpirun) -np %d'!", PROC_NUM);
		MPI_Finalize ();
		return 1;
	}

	if (rank == 0) {
		int id, *src_mat;

		if ((id = get_shmid('r')) < 0) {
			perror("Shared memory allocate failed.\n");
			exit(1);
		}

		if ((src_mat = (int *)shmat(id, NULL, 0)) == (void *) - 1) {
			perror("Shared memory fetch failed.\n");
			exit(1);
		}

		set_random(src_mat, MAT_SIZE * MAT_SIZE);
	}

	if (rank == 1) {
		if (get_shmid('w') < 0) {
			perror("Shared memory allocate failed.\n");
			exit(1);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	int src_id = get_shmid('R'), des_id = get_shmid('W');printf("%d %d\n",src_id,des_id);
	int *src_mat = (int*)shmat(src_id, NULL, 0), *des_mat = (int*)shmat(des_id, NULL, 0);

	double r0 = MPI_Wtime();
	int row = MAT_SIZE / PROC_NUM, col = MAT_SIZE;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			MPI_Send(src_mat + col * (i + rank * row) + j, 1, MPI_INT, j / row, col * (i + rank * row) + j, MPI_COMM_WORLD);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			MPI_Recv(des_mat + col * (i + rank * row) + j, 1, MPI_INT, j / row, j * col + rank * row + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	double r1 = MPI_Wtime();
	if (rank == 0)
		printf("MPI_Alltoall time (sec) %lf.\n", r1 - r0);

	/* Check the result after transposition */
	if (rank == 0)
		Test(src_mat, des_mat, MAT_SIZE);

	if (shmdt(src_mat) == -1 || shmdt(des_mat) == -1) {
		perror("shmdt");
		exit(1);
	}

	MPI_Finalize ();
	return 0;
}
