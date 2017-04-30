#ifndef TIMING_H
#define TIMING_H

/* timing.h provides a class to help record the running time in GPU */

#include <cstdio>

/* Define a class to help record running time and print it out */
class Timing {
private:
	cudaEvent_t begin, end;

public:
	Timing() {
		cudaEventCreate(&begin);
		cudaEventCreate(&end);
	}
	/*
	 * beginTiming() records the beginning of computing on GPU
	 * endTiming() records the ending of computing on GPU
	 */
	void beginTiming() {
		cudaEventRecord(begin);
	}
	void endTiming() {
		cudaEventRecord(end);
		cudaEventSynchronize(end);
	}
	/* printTime() prints the running time out */
	void printTime() {
		float time;

		cudaEventElapsedTime(&time, begin, end);
		printf("The process takes %.6f milliseconds.\n", time);
	}
};

#endif
