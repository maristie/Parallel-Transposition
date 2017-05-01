## Environment
MPICH

## Commands
```bash
make		# compile
make run	# run
```

## Design
16 processes divide a 8192\*8192 matrix to 16 shares. Then each process allocates memories and initializes. Send submatrices by *MPI_Alltoall* to implement transposition and finally transpose each submatrix internally.

### Initialization
Each element of matrix has been assigned a unique value.

### Versions
Divided by rows or columns differ in efficiency.

### Optimization
Use -O3 in Makefile.
