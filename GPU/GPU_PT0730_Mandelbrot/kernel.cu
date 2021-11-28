
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

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
    int i = 0;
    float curX, curY;
    curY = *startY;
    while (curY < endY) {
        int j = 0;
        curX = *startX;
        while (curX < endX) {
            pointApproximation(&curX, &curY, maxIter, approximation + (i * (int)(*width / *step + 0.5)) + j++);
            curX += *step;
        }
        curY += *step;
        i++;
    }
}

__global__ void Func(float** startX, float** startY, int** appro, float* step, int* maxIter, float* width, int* numberThreads)
{
    int i = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * 8 + threadIdx.x + threadIdx.y * 8;
    //traverse(startX + i, startY + i, (startX[i] + *width), (startY[i] + *width), step, maxIter, appro + (8*((i/8)*4))+2*(i%8), width);
    appro[1][1] = 2137;
    //if (appro[1][1] < 0)
        printf("%d ",appro[1][1]);
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

    //float* realPoints;
    //realPoints = (float*)malloc(sizeof(float) * numberThreads*numberThreads);

    float** realPoints;
    realPoints = (float**)malloc(sizeof(float*) * numberThreads);
    for (int i = 0; i < numberThreads; i++)
    {
        realPoints[i] = (float*)malloc(sizeof(float) * numberThreads);
    }

    float** imagPoints;
    imagPoints = (float**)malloc(sizeof(float*) * numberThreads);
    for (int i = 0; i < numberThreads; i++)
    {
        imagPoints[i] = (float*)malloc(sizeof(float) * numberThreads);
    }

    int** approximations;
    approximations = (int**)malloc(sizeof(int*) * (width / step) * numberThreads);
    for (int i = 0; i < (width / step) * numberThreads; i++)
    {
        approximations[i] = (int*)malloc(sizeof(int) * (width / step) * numberThreads);
    }


    //float* imagPoints;
    //imagPoints = (float*)malloc(sizeof(float) * numberThreads*numberThreads);

    //int* approximations;
    //approximations = (int*)malloc(sizeof(int) * (width/step)* (width / step)* numberThreads * numberThreads);

    for (int i = 0; i < (width / step) * numberThreads; i++) 
    {
        for (int j = 0; j < (width / step) * numberThreads; j++)
        {
            approximations[i][j] = -2;
        }

    }

    for (int i = 0; i < numberThreads; i++)
    {
        for (int j = 0; j < numberThreads; j++)
        {
            realPoints[i][j] = -2.0 + width * i;
            imagPoints[i][j] = -2.0 + width * j;
        }
    }

    float** realPoints_c, **imagPoints_c;
    int** approximations_c;
    float* width_c;
    int* max_c;
    float* step_c;
    int* numberThreads_c;

    size_t pitchR;
    cudaMallocPitch((void**)&realPoints_c, &pitchR, numberThreads, numberThreads);
    //cudaMalloc((void**)&realPoints_c, sizeof(float*) * numberThreads);
    //for (int i = 0; i < numberThreads; i++)
    //{
    //    cudaMalloc((void**)&realPoints_c[i], sizeof(float) * numberThreads);
    //}
    size_t pitchI;
    cudaMallocPitch((void**)&imagPoints_c, &pitchI, numberThreads, numberThreads);
    //for (int i = 0; i < numberThreads; i++)
    //{
    //    cudaMalloc((void**)&imagPoints_c[i], sizeof(float) * numberThreads);
    //}
    size_t pitchA;
    cudaMallocPitch((void**)&approximations_c, &pitchA, (width / step) * numberThreads, (width / step) * numberThreads);
    //for (int i = 0; i < (width / step) * numberThreads; i++)
    //{
    //    cudaMalloc((void**)&approximations_c[i], sizeof(int) * (width / step) * numberThreads);
    //}

    cudaMalloc((void**)&width_c, sizeof(float));
    cudaMalloc((void**)&max_c, sizeof(int));
    cudaMalloc((void**)&step_c, sizeof(float));
    cudaMalloc((void**)&numberThreads_c, sizeof(int));

    cudaMemcpy2D(realPoints_c, pitchR, realPoints, sizeof(float*) * numberThreads, numberThreads, numberThreads, cudaMemcpyHostToDevice);
    cudaMemcpy2D(imagPoints_c, pitchI, imagPoints, sizeof(float*) * numberThreads, numberThreads, numberThreads, cudaMemcpyHostToDevice);
    cudaMemcpy(width_c, &width, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(max_c, &max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(step_c, &step, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy2D(approximations_c, pitchA, approximations, sizeof(int*) * (width / step) * numberThreads, (width / step) * numberThreads, (width / step) * numberThreads, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(4, 4);
    dim3 numBlocks(numberThreads / threadsPerBlock.x, numberThreads / threadsPerBlock.y);

    Func << <numBlocks, threadsPerBlock >> > (realPoints_c, imagPoints_c, approximations_c, step_c, max_c, width_c, numberThreads_c);

    cudaMemcpy(approximations, approximations_c, sizeof(int*) * (width / step) * numberThreads, cudaMemcpyDeviceToHost);

    for (int i = 0; i < (width / step) * numberThreads; i++)
    {
        for (int j = 0; j < (width / step) * numberThreads; j++)
        {
            if (approximations[i][j] >= 0)
                cout << approximations[i] << "\t";
            else
                cout << "  \t";
        }
        cout << endl;
    }
    return 0;
}


