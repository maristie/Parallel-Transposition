## Environment
MPICH

## Commands
```bash
mpicc -o trans trans.c	// add -std=c99 for older version of gcc
mpirun -np 16 ./trans
```

## Design
Divide a matrix by columns and each process takes responsibility for memory allocation and value initialization. Here each element is assigned a unique value for the purpose of verifying afterwards. For each process its submatrix will be divided further to blocks and communicate with *MPI_Alltoall* to transpose submatrices. Finally transpose submatrices internally.

## Trans 1 & 2
Trans 1 is designed just like above. Trans 2 is based on shared memory mechanism in Linux, in which we assume that all elements are stored and initialized in a block of shared memory.
