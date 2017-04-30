#ifndef TEST_H
#define TEST_H

/* test.h is to test whether a matrix is successfully transposed */

#include <cstdio>

/* Return 0 when everything is OK, otherwise return 1 */
int test(int *addr_a, int *addr_b) {
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (addr_b[j * N + i] != addr_a[i * N + j])
				return 1;
	return 0;
}

/* Wrapped function */
void Test(int *addr_a, int *addr_b) {
	if (test(addr_a, addr_b) == 0)
		printf("Result: Correct.\n");
	else
		printf("Result: Wrong.\n");
}

#endif
