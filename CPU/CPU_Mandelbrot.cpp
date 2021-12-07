// CPU_Mandelbrot.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <fstream>
#include <string>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <thread>
#include <time.h>
#include <windows.h>
#define AREA_UNSINED_LIMIT 2

using namespace std;

void pointApproximation(float* realPart, float* imagPart, int* maxIter, int* approximation);
void traverse(float startX, float startY, float endX, float endY, float* step, int* maxIter, int* approximation, float* width);
float** splitArea(int* threadCount);
void print_in_human_readable_form(float** table, int x, int y, int n);

#pragma region Kod do pomiaru czasu
double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart) / 1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart - CounterStart) / PCFreq;
}
#pragma endregion

int main()
{
    cout << "Mandelbrot CPU" << "\r\n";

    float realPoint = -AREA_UNSINED_LIMIT;
    float imagPoint = -AREA_UNSINED_LIMIT;

    int max;
    float step;
    int appro = 0;
    int threadCount;

    cout << "Podaj maksymalna ilosc iteracji: ";
    cin >> max;

    cout << "Podaj dokladnosc liczby zmiennoprzecinkowej: ";
    cin >> step;
    
    cout << "Podaj ilosc watkow: ";
    cin >> threadCount;

    //Time measurement
    double ms = 0;
    StartCounter();

    //Helpful values
    float width = 2 * AREA_UNSINED_LIMIT;
    int size = (int)((width / step) + 0.5);

    //Result array
    int* approximations = new int[(size + 1) * (size + 1)]; //DO NOT TOUCH

    //Complex plane coordinates
    float startX = -2.0;
    float startY = 0;
    float endX = 2.0;
    float endY = 0;

    //Most of the rectangles height: (n-1) rectangles of this height and 1 rectangle of remaining height
    int most_of_rectangles_height = size / threadCount;

    //Most of the rectangles size ratio
    float most_of_rectangles_size_ratio = size / (float)most_of_rectangles_height;

    //Thread array
    thread* threads = new thread[threadCount];

    //Prepare n-1 threads
    for (int i = 0; i < threadCount - 1; i++)
    {
        //Prepare correct parameters
        startY = (float)-AREA_UNSINED_LIMIT + 2 * (float)AREA_UNSINED_LIMIT * i / most_of_rectangles_size_ratio;
        endY = (float)-AREA_UNSINED_LIMIT + 2 * (float)AREA_UNSINED_LIMIT * (i + 1) / most_of_rectangles_size_ratio;

        //Run thread with correct parameters
        threads[i] = thread(traverse, startX, startY, endX, endY, &step, &max, approximations + (int)(size * size * i / most_of_rectangles_size_ratio + 0.5), &width);
    }

    //Prepare last thread
    //Prepare correct parameters
    startY = (float)-AREA_UNSINED_LIMIT + 2 * (float)AREA_UNSINED_LIMIT * (threadCount - 1) / most_of_rectangles_size_ratio;
    endY = 2;
    //Run last thread with correct parameters
    threads[threadCount - 1] = thread(traverse, startX, startY, endX, endY, &step, &max, approximations + (int)(size * size * (threadCount - 1) / most_of_rectangles_size_ratio + 0.5), &width);
    
    //Join the threads
    for (int i = 0; i < threadCount; i++) threads[i].join();

    //Time measurement
    ms += GetCounter();

    //Display
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            cout << setw(2) << approximations[(int)(i * size + j)];
        }
        cout << endl;
    }

    cout << "\r\n\r\n" << ms << "ms" << "\r\n";
    
    //Because of size of resulting array it is best to save data into file
    //std::fstream file("mandelbrot.pgm", std::fstream::out);
    //file << "P2\n" << size << " " << size << "\n" << max << "\n";
    //std::string line, value;

    //line = "";
    //for (int i = 0; i < size * size; i++)
    //{
    //    value = to_string(approximations[(int)(i)]);
    //    if(line.length() + value.length() > 69)
    //    {
    //        file << line << "\n";
    //        line = "";
    //    }
    //    line += value + " ";
    //}

    //file << line;

    //file.close();

    //Cleanup
    delete[] approximations;

    //Exit
    return 0;
}

float** splitArea(int* threadCount)
{
    int totalLength = AREA_UNSINED_LIMIT << 1;
    int rowCount = int(sqrt(*threadCount));

    if (rowCount * rowCount < *threadCount) rowCount++;
    float areaSide = float(totalLength) / rowCount;
    float minPoint = float(-AREA_UNSINED_LIMIT);
    float** dims = new float*[2];
    dims[0] = new float[rowCount + 2];
    dims[1] = new float;
    int inputIdx = 1;

    while (minPoint < AREA_UNSINED_LIMIT) {
        dims[0][inputIdx++] = minPoint;
        minPoint += areaSide;
    }
    dims[0][inputIdx] = AREA_UNSINED_LIMIT;
    
    if (rowCount * rowCount > * threadCount)
    {
        dims[1][0] = 1.0;
    } 
    else dims[1][0] = 0.0;

    dims[0][0] = rowCount;

    return dims;
}

void print_in_human_readable_form(float** table, int x, int y, int n)
{
    float startX = table[0][x];
    float startY = table[0][y];
    float endX = table[0][n];
    float endY = table[0][y + 1];

    cout << "[" << startY << ", " << startX << "] [" << endY << ", " << endX << "]" << "\r\n";
}

void pointApproximation(float* realPart, float* imagPart, int* maxIter, int* approximation)
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

void traverse(float startX, float startY, float endX, float endY, float* step, int* maxIter, int* approximation, float* width)
{
    int i = 0;
    float curX, curY;
    curY = startY;
    while (curY < endY) {
        int j = 0;
        curX = startX;
        while (curX < endX) {
            pointApproximation(&curX, &curY, maxIter, approximation + (i * (int)(*width / *step + 0.5)) + j++);
            curX += *step;
        }
        curY += *step;
        i++;
    }
}
