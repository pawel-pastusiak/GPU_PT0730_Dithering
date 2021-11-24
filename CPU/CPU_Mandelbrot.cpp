// CPU_Mandelbrot.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <math.h>
#define AREA_UNSINED_LIMIT 2

using namespace std;

void pointApproximation(float* realPart, float* imagPart, int* maxIter, int* approximation);
void traverse(float* startX, float* startY, float* endX, float* endY, float* step, int* maxIter, int* approximation);
float** splitArea(int* threadCount);

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
    for (int i = 0; i < threadCount - 2; i++)
    {
        cout << "{" << table[0][Y] << ", " << table[0][X++] << "}\t";
        if ((i + 1) % count == 0)
        {
            X = 1;
            cout << "{" << table[0][Y] << ", " << table[0][count + 1] << "}";
            cout << endl << endl;
            Y++;
        }
    }
    if (table[1][0] < 0.5)
    {
        cout << "{" << table[0][count] << ", " << table[0][count - 1] << "}\t{" << table[0][count] << ", " << table[0][count] << "}";
        cout << "\t{" << table[0][count] << ", " << table[0][count + 1] << "}" << endl << endl;

    }
    else
    {
        cout << "{" << table[0][Y] << ", " << table[0][X] << "}\t{" << table[0][Y] << ", ";
        if (X++ < count)
        {
            if (Y == count) X = count - 1;
            else X = count + 1;
        }
        cout << table[0][X] << "}" << endl << endl;
        if (Y != count) X = 1;
        cout << "{" << table[0][count] << ", " << table[0][X] << "}\t{" << table[0][count] << ", " << table[0][count + 1] << "}" << endl << endl;
    }

    //pointApproximation(&realPoint, &imagPoint, &max, &appro);
    //cout << appro;
    //traverse(&realPoint, &imagPoint, new float(1.0), new float(1.0), &step, &max, &appro);
    //cout << appro;
    cin >> max;
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
    dims[0] = new float(rowCount + 2);
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
