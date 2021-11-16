
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

void pointApproximation(float* realPart, float* imagPart, int* maxIter, int* approximation);

int main()
{
    float realPoint = 0.2;
    float imagPoint = 0.2;
    int max = 100;
    int appro = 0;
    pointApproximation(&realPoint, &imagPoint, &max, &appro);
    cout << appro;

    return 0;
}

void pointApproximation(float* realPart, float* imagPart, int *maxIter, int *approximation)
{
    int i = 0;
    float zReal = 0;
    float zImag = 0;
    float zTempReal = 0;
    float zTempImag = 0;

    while (i < *maxIter && (zReal * zReal + zImag * zImag < 4))
    {
        zTempReal = zReal * zReal - zImag * zImag;
        zTempImag = 2 * zReal * zImag;

        zReal = zTempReal + *realPart;
        zImag = zTempImag + *imagPart;


        i++;
    }

    *approximation = i;
}