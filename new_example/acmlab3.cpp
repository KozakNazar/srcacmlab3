// don't forget to use compilation key for Linux: -lm
/*******************************************************************************************
* N.Kozak // Lviv'2021 // ACM // Algorithm Flow Graph(compute x = A*B*C*D*E*F*G*H by SSE2) *
*    file: acmlab3.cpp                                                                     *
********************************************************************************************/
//-fno-tree-vectorize
//gcc -O3 -no-tree-vectorize 
//gcc -O3 -ftree-vectorizer-verbose=6 -msse4.1 -ffast-math 
//__attribute__((optimize("no-tree-vectorize")))
//extern "C" __attribute__ ((optimize("no-tree-vectorize")))

#include <stdio.h>
#include <stdlib.h>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#include <intrin.h> // Windows
#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <x86intrin.h> // Linux
#else
#   error "Unknown compiler"
#endif
#include <math.h> 
#include <time.h>

#define NANOSECONDS_PER_SECOND_NUMBER 1000000000

#define DATA_TYPE                       double
#define DATA_TYPE_PTR_  DATA_TYPE       *
#define DATA_TYPE_PTR   DATA_TYPE_PTR_  volatile

#define A  1.
#define B  2.
#define C  3.
#define D  4.
#define E  5.
#define F  6.
#define G  7.
#define H  8.

#define dAB      dArr
#define dA       dAB
#define dB      (dAB + 1)
#define dCD     (dArr + 2)
#define dC      (dCD)
#define dD      (dCD + 1)
#define dEF     (dArr + 4)
#define dE       dEF
#define dF      (dEF + 1)
#define dGH     (dArr + 6)
#define dG       dGH
#define dH      (dGH + 1)
#define dResult (dArr + 8)
#define dX       dResult

#pragma GCC push_options
#pragma GCC optimize ("no-unroll-loops")
//#pragma clang attribute push (__attribute__((target("no-unroll-loops"))), apply_to=function)

#define REPEAT_COUNT 1000000
#define REPEATOR(count, code) \
for (unsigned int indexIteration = (count); indexIteration--;){ code; }

double getCurrentTime() {
	clock_t time = clock();
	if (time != (clock_t)-1) {
		return ((double)time / (double)CLOCKS_PER_SEC);
	}
	return 0.; // else 	
}

//#pragma clang attribute push (__attribute__((target("no-sse2"))), apply_to=function)
#pragma GCC push_options
#pragma GCC target ("no-sse2")
//__attribute__((__target__("no-sse2")))
void run_native(DATA_TYPE_PTR const dArr) {
	*dX *= *dA;
	*dX *= *dB;
	*dX *= *dC;
	*dX *= *dD;
	*dX *= *dE;
	*dX *= *dF;
	*dX *= *dG;
	*dX *= *dH;
}
#pragma GCC pop_options
//#pragma clang attribute pop

void run_SSE2(DATA_TYPE_PTR const dArr) {
	__m128d d__A_B__;
	__m128d d__AC_BD__;
	__m128d d__ACE_BDF__;
	__m128d d__ACEG_BDFH__;
	__m128d d__ACEGBDFH__;

	// (etap 1)			
	d__A_B__ = _mm_load_pd(dAB);
	d__AC_BD__ = _mm_load_pd(dCD);
	d__AC_BD__ = _mm_mul_pd(d__A_B__, d__AC_BD__);

	// (etap 2)			
	d__ACE_BDF__ = _mm_load_pd(dEF);
	d__ACE_BDF__ = _mm_mul_pd(d__AC_BD__, d__ACE_BDF__);

	// (etap 3)			
	d__ACEG_BDFH__ = _mm_load_pd(dGH);
	d__ACEG_BDFH__ = _mm_mul_pd(d__ACE_BDF__, d__ACEG_BDFH__);

	// (etap 4)			
	d__ACEGBDFH__ = _mm_unpackhi_pd(d__ACEG_BDFH__, d__ACEG_BDFH__);
	d__ACEGBDFH__ = _mm_mul_sd(d__ACEGBDFH__, d__ACEG_BDFH__);
		
	_mm_store_sd(dResult, d__ACEGBDFH__);
}


void printResult(char* const title, DATA_TYPE_PTR const dArr, unsigned int runTime) {
	printf("%s:\r\n", title);
	printf("x = A * B * C * D * E * F * H * G\r\n");
	printf("A=%f, B=%f, C=%f, D=%f, E=%f, F=%f, H=%f, G=%f;\r\n", *dA, *dB, *dC, *dD, *dE, *dF, *dG, *dH);
	printf("x = %1.0f;\r\n", *dX);
	printf("run time: %dns\r\n\r\n", runTime);
}

int main() {
	DATA_TYPE_PTR const dArr = (DATA_TYPE_PTR_ const)_mm_malloc(63 * sizeof(DATA_TYPE), 16);
	if (!dArr) {
		return 0;
	}

	*dA = A;
	*dB = B;
	*dC = C;
	*dD = D;
	*dE = E;
	*dF = F;
	*dG = G;
	*dH = H;

	double startTime, endTime;

	// native (only x86, if auto vectorization by compiler is off) 
	startTime = getCurrentTime();
	REPEATOR(REPEAT_COUNT,
		run_native(dArr);
	)
		endTime = getCurrentTime();
	printResult((char*)"x86",
		dArr,
		(unsigned int)((endTime - startTime) * (NANOSECONDS_PER_SECOND_NUMBER / REPEAT_COUNT)));

	*dX = 0;

	// SSE2
	startTime = getCurrentTime();
	REPEATOR(REPEAT_COUNT,
		run_SSE2(dArr);
	)
		endTime = getCurrentTime();
	printResult((char*)"SSE2",
		dArr,
		(unsigned int)((endTime - startTime) * (NANOSECONDS_PER_SECOND_NUMBER / REPEAT_COUNT)));

	_mm_free(dArr);

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
	//printf("Press Enter to continue . . .");
	//(void)getchar();
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)    
	system("pause");
#else
#endif

	return 0;
}

#pragma GCC pop_options
//#pragma clang attribute pop
