
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

//__global__ void pointApproximation(float* realPart, float* imagPart, int* maxIter, int* approximation);
//__global__ void traverse(float* startX, float* startY, float* endX, float* endY, float* step, int* maxIter, int* approximation, float* width);

__device__ void pointApproximation(float* realPart, float* imagPart, int* maxIter, int* approximation)
{
    *approximation = 0;
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

__device__ void traverse(float* startX, float* startY, float endX, float endY, float* step, int* maxIter, int* approximation, float* width)
{
    printf("aaaaa");
    *approximation = 2137;
    int i = 0;
    float curX, curY;
    curX = *startX;
    while (curX < endX) {
        int j = 0;
        curY = *startY;
        while (curY < endY) {
            //pointApproximation(&curX, &curY, maxIter, approximation + (i * (int)(*width / *step + 0.5)) + j++);
            
            //cout << endl << i++ << ": " << curX << " x " << curY << ": " << *approximation;
            curY += *step;
        }
        curX += *step;
        i++;
    }
}

__global__ void Func(float* startX, float* startY, int* appro, float* step, int* maxIter, float* width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //traverse(startX+i, startY+i, (startX[i] + *width), (startY[i] + *width), step, maxIter, appro + i, width);
    pointApproximation(startX + i, startY + i, maxIter, appro + i);
    //*appro = 2137;
}

int main()
{
    int max = 0;
    float step = 0;

    cout << "Prosze podac dokladnosc liczby zmiennoprzecinkowej: ";
    cin >> step;
    cout << "\nProsze podac maksymalna liczbe powtorzen funkcji sprawdzajacej przynaleznosc punktu do zbioru: ";
    cin >> max;

    int numberThreads = 8;
    float width = 4.0 / numberThreads;

    float* realPoints;
    realPoints = (float*)malloc(sizeof(float) * numberThreads*numberThreads);

    float* imagPoints;
    imagPoints = (float*)malloc(sizeof(float) * numberThreads*numberThreads);

    int* approximations;
    approximations = (int*)malloc(sizeof(int) * (width/step)* (width / step)* numberThreads * numberThreads);

    for (int i = 0; i < numberThreads; i++)
    {
        for (int j = 0; j < numberThreads; j++)
        {
            realPoints[i * numberThreads + j] = -2.0 + width * i;
            imagPoints[i * numberThreads + j] = -2.0 + width * j;
        }
    }

    float* realPoints_c, *imagPoints_c;
    int* approximations_c;

    cudaMalloc((void**)&realPoints_c, sizeof(float) * numberThreads*numberThreads);
    cudaMalloc((void**)&imagPoints_c, sizeof(float) * numberThreads * numberThreads);
    cudaMalloc((void**)&approximations_c, sizeof(int) * (width / step) * (width / step) * numberThreads * numberThreads);

    cudaMemcpy(realPoints_c, realPoints, sizeof(float) * numberThreads*numberThreads, cudaMemcpyHostToDevice);
    cudaMemcpy(imagPoints_c, imagPoints, sizeof(float) * numberThreads * numberThreads, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(4, 4);
    dim3 numBlocks(numberThreads * numberThreads / threadsPerBlock.x, numberThreads * numberThreads / threadsPerBlock.y);

    Func << <numBlocks, threadsPerBlock >> > (realPoints_c, imagPoints_c, approximations_c, &step, &max, &width);

    cudaMemcpy(approximations, approximations_c, sizeof(int) * (width / step) * (width / step) * numberThreads * numberThreads, cudaMemcpyDeviceToHost);

    for (int i = 0; i < (width / step) * (width / step) * numberThreads * numberThreads; i++)
    {
        if (i % (int)((width / step) * numberThreads + 0.5) == 0)
            cout << endl;
        cout << approximations[i] << "\t";
    }

    //cudaMemcpy(realPoints, realPoints_c, sizeof(float) * numberThreads * numberThreads, cudaMemcpyDeviceToHost);

    //for (int i = 0; i < numberThreads * numberThreads; i++)
    //{
    //    if (i % numberThreads == 0)
    //        cout << endl;
    //    cout << "{" << realPoints[i] << "," << imagPoints[i] << "}\t";
    //}

    //int* den;
    //den = (int*)malloc(sizeof(int) * 17);
    //int* d_a;
    //cudaMalloc((void**)&d_a, sizeof(int) * 17);
    //cudaMemcpy(d_a, den, sizeof(int) * 17, cudaMemcpyHostToDevice);

    //Func << <1, 17 >> > (d_a);

    //cudaMemcpy(den, d_a, sizeof(int) * 17, cudaMemcpyDeviceToHost);

    //for (int i = 0; i < 17; i++)
    //{
    //    cout << den[i] << " ";
    //}

    //float realPoint = 0.2;
    //float imagPoint = 0.2;
    //int max = 0;
    //float step = 0;
    //int appro = 0;

    //cout << "Prosze podac dokladnosc liczby zmiennoprzecinkowej: ";
    //cin >> step;
    //cout << "\nProsze podac maksymalna liczbe powtorzen funkcji sprawdzajacej przynaleznosc punktu do zbioru: ";
    //cin >> max;

    //pointApproximation(&realPoint, &imagPoint, &max, &appro);
    ////cout << appro;
    //traverse(&realPoint, &imagPoint, new float(1.0), new float(1.0), &step, &max, &appro);

    return 0;
}


