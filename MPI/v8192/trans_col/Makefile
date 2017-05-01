CC = mpicc

trans_col : trans_col.c
	$(CC) -O3 -o $@ $<

.PHONY : run

run : ./trans_col
	mpirun -np 16 ./trans_col
