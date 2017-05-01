#ifndef RANDOM_H
#define RANDOM_H

/* random.h is to set random numbers for each element in a matrix */

#include <cstdio>
#include <ctime>
#include <cstdlib>

void set_random(int *matrix) {
	srand((unsigned)time(0));
	for (int i = 0; i < ELEMENT_NUM; i++)
		matrix[i] = rand();
}

#endif
