// CPU_Mandelbrot.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <math.h>
#define AREA_UNSINED_LIMIT 2

using namespace std;

void pointApproximation(float* realPart, float* imagPart, int* maxIter, int* approximation);
void traverse(float* startX, float* startY, float* endX, float* endY, float* step, int* maxIter, int* approximation);
float** splitArea(int* threadCount);
void print_in_human_readable_form(float** table, int x, int y, int n);

int main()
{
    cout << "Mandelbrot CPU" << "\r\n";

    float realPoint = -AREA_UNSINED_LIMIT;
    float imagPoint = -AREA_UNSINED_LIMIT;

    int max;
    float step;
    int appro = 0;
    int threadCount;

    /*cout << "Podaj maksymalna ilosc iteracji: ";
    cin >> max;

    cout << "Podaj dokladnosc liczby zmiennoprzecinkowej: ";
    cin >> step;
    */
    cout << "Podaj ilosc watkow: ";
    cin >> threadCount;

    cout << "DEBUG:" << endl;
    
    int Y = 1;
    int X = 1;
    float** table = splitArea(&threadCount);
    int count = int(table[0][0]);
    //cout << count << " " << table[0][0] << "\r\n\r\n";
    /*
    for (int i = 0; i < count; i++)
        for (int j = 0; j < ((i == count - 1) ? count - 1 : count); j++)
            print_in_human_readable_form(table, i+1, j+1, i * count + j);
   */
    

    for (int i = 0; i < threadCount - 2; i++)
    {
        print_in_human_readable_form(table, X, Y, X + 1);
        X++;
        if ((i + 1) % count == 0)
        {
            X = 1;
            
            Y++;
        }
    }
    if (table[1][0] < 0.5)
    {
        print_in_human_readable_form(table, count - 1, count, count);
        print_in_human_readable_form(table, count, count, count + 1);

    }
    else
    {
        int temp = 0;
        if (X < count && Y == count) temp = count - 1;
        else temp = count + 1;

        print_in_human_readable_form(table, X, Y, temp);

        if (Y != count) temp = 1;
        
        print_in_human_readable_form(table, temp, count, count + 1);
    }

    //pointApproximation(&realPoint, &imagPoint, &max, &appro);
    //cout << appro;
    //traverse(&realPoint, &imagPoint, new float(1.0), new float(1.0), &step, &max, &appro);
    //cout << appro;
    //cin >> max;
    delete[] table[0];
    delete table[1];
    delete[] table;
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

void traverse(float* startX, float* startY, float* endX, float* endY, float* step, int* maxIter, int* approximation)
{
    int i = 0;
    float curX, curY;
    curX = *startX;
    while (curX < *endX) {
        curY = *startY;
        while (curY < *endY) {
            pointApproximation(&curX, &curY, maxIter, approximation);
            //cout << endl << i++ << ": " << curX << " x " << curY << ": " << *approximation;
            curY += *step;
        }
        curX += *step;
    }
}
