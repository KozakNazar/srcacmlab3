// don't forget to use compilation key for Linux: -lm
/**********************************************************************************************************
* N.Kozak // Lviv'2018 // ACM // Algorithm Flow Graph(by solving a quadratic equation example using SSE2) *
*    file: acmlab3.c                                                                                      *
***********************************************************************************************************/
//-fno-tree-vectorize
//gcc -O3 -no-tree-vectorize 
//gcc -O3 -ftree-vectorizer-verbose=6 -msse4.1 -ffast-math 
//__attribute__((optimize("no-tree-vectorize")))
//extern "C" __attribute__ ((optimize("no-tree-vectorize")))

#include <stdio.h>
#include <stdlib.h>
//#include <x86intrin.h> // Linux
#include <intrin.h> // Windows
#include <math.h> 
#include <time.h>

#define A  0.33333333
#define B  0.
#define C -3.

#pragma GCC push_options
#pragma GCC optimize ("no-unroll-loops")

#define REPEAT_COUNT 1000000
#define REPEATOR(count, code) \
for (unsigned int indexIteration = (count); indexIteration--;){ code; }
#define TWO_VALUES_SELECTOR(variable, firstValue, secondValue) \
	(variable) = indexIteration % 2 ? (firstValue) : (secondValue);

double getCurrentTime(){
	clock_t time = clock();
	if (time != (clock_t)-1) { 
		return ((double)time / (double)CLOCKS_PER_SEC); 
	}
	return 0.; // else 	
}

#pragma GCC push_options
#pragma GCC target ("no-sse2")
//__attribute__((__target__("no-sse2")))
void run_native(double * const dArr){
	double * const dAC = dArr;
	double * const dA = &dAC[0];
	double * const dC = &dAC[1];
	double * const dB = &dArr[2];
	double * const dResult = &dArr[4];
	double * const dX1 = &dResult[1];
	double * const dX2 = &dResult[0];

	REPEATOR(REPEAT_COUNT, 
	    TWO_VALUES_SELECTOR(*dA, 4., A); 
	    TWO_VALUES_SELECTOR(*dB, 3., B); 
	    TWO_VALUES_SELECTOR(*dC, 1., C);
	    double vD = sqrt((*dB)*(*dB) - 4.*(*dA)*(*dC));
	    (*dX1) = (-(*dB) + vD) / (2.*(*dA));
	    (*dX2) = (-(*dB) - vD) / (2.*(*dA));
	)
}
#pragma GCC pop_options

void run_SSE2(double * const dArr){
	double * const dAC = dArr;
	double * const dA = &dAC[0];
	double * const dC = &dAC[1];
	double * const dB = &dArr[2];
	double * const dResult = &dArr[4];
	double * const dX1 = &dResult[1];
	double * const dX2 = &dResult[0];

	__m128d r__zero_zero, r__c_a, r__uORb_b, r__2cORbOR2a_2a, 
	r__zero_bb, r__sqrtDiscriminant_zero, r_result;

	r__zero_zero = _mm_set_pd(0., 0.); // init

	REPEATOR(REPEAT_COUNT, 
	    TWO_VALUES_SELECTOR(*dA, 4., A); 
	    TWO_VALUES_SELECTOR(*dB, 3., B); 
	    TWO_VALUES_SELECTOR(*dC, 1., C);
	    r__c_a = _mm_load_pd(dAC);
		// r__uORb_b = _mm_load_pd1(dB);
	    r__uORb_b = _mm_load1_pd(dB);
        // b b		
	    r__uORb_b = _mm_unpacklo_pd(r__uORb_b, r__uORb_b);
        // (etap 1)		
	    r__2cORbOR2a_2a = _mm_add_pd(r__c_a, r__c_a); 
        // b 2c		
	    r_result = _mm_unpackhi_pd(r__2cORbOR2a_2a, r__uORb_b); 
		// b 2a
	    r__2cORbOR2a_2a = _mm_unpacklo_pd(r__2cORbOR2a_2a, r__uORb_b);
        // bb 4ac (etap 2)		
	    r_result = _mm_mul_pd(r_result, r__2cORbOR2a_2a); 
	    r__zero_bb = _mm_unpackhi_pd(r_result, r__zero_zero);
		// zero Discriminant (etap 3)
	    r_result = _mm_sub_sd(r__zero_bb, r_result);
        // zero sqrtDiscriminant (etap 4)		
	    r_result = _mm_sqrt_sd(r_result, r_result);   
	    r__sqrtDiscriminant_zero = _mm_shuffle_pd(r_result, r_result, 1);
		// sqrtDiscriminant -sqrtDiscriminant (etap 5)
	    r_result = _mm_sub_sd(r__sqrtDiscriminant_zero, r_result); 
        // (etap 6)		
	    r_result = _mm_sub_pd(r_result, r__uORb_b);
        // 2a 2a		
	    r__2cORbOR2a_2a = _mm_unpacklo_pd(r__2cORbOR2a_2a, r__2cORbOR2a_2a); 
		// (etap 7)
	    r_result = _mm_div_pd(r_result, r__2cORbOR2a_2a); 
	    _mm_store_pd(dResult, r_result);
	)
}

void printResult(char * const title, double * const dArr, unsigned int runTime){
	double * const dAC = dArr;
	double * const dA = &dAC[0];
	double * const dC = &dAC[1];
	double * const dB = &dArr[2];
	double * const dResult = &dArr[4];
	double * const dX1 = &dResult[1];
	double * const dX2 = &dResult[0];

	printf("%s:\r\n", title);
	printf("%fx^2 + %fx + %f = 0;\r\n", *dA, *dB, *dC);
	printf("x1 = %1.0f; x2 = %1.0f;\r\n", *dX1, *dX2);
	printf("run time: %dns\r\n\r\n", runTime);
}

int main() {
	double * const dArr = (double *)_mm_malloc(6 * sizeof(double), 16);

	double * const dAC = dArr;
	double * const dA = &dAC[0];
	double * const dC = &dAC[1];
	double * const dB = &dArr[2];
	double * const dResult = &dArr[4];
	double * const dX1 = &dResult[1];
	double * const dX2 = &dResult[0];

	double startTime, endTime;

	// native (only x86, if auto vectorization by compiler is off) 
	startTime = getCurrentTime();
	run_native(dArr);
	endTime = getCurrentTime();
	printResult((char*)"x86", 
	dArr, 
	(unsigned int)((endTime - startTime) * (1000000000 / REPEAT_COUNT)));

	// SSE2
	startTime = getCurrentTime();
	run_SSE2(dArr);
	endTime = getCurrentTime();
	printResult((char*)"SSE2",
	dArr, 
	(unsigned int)((endTime - startTime) * (1000000000 / REPEAT_COUNT)));

	_mm_free(dArr);

	printf("Press any key to continue . . .");
	getchar();
	return 0;
}

#pragma GCC pop_options
