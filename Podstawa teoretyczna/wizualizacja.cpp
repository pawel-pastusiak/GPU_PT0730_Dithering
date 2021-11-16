#include <iostream>
#include <iomanip>

int main() 
{
	int precision, iterating[] = { 1024, 2048, 4096, 8192, 16384 };
	float x, y, z;
	std::cout << "Wizualizacja predkosci obliczen dla kolejnych poteg 2, zaczynajac od 1024:";
	for (int i = 0; i < 5; i++)
	{
		x = 0.0;
		z = 1.0 / iterating[i];
		precision = 6 + i / 2;
		std::cout << std::endl << iterating[i]  << std::endl;
		
		while (x < 1) 
		{
			x += z;
			y = 0.0;
			
			while (y < 1) 
			{
				y += z;
				// printing a value with suspected precision
				std::cout << std::setprecision(precision) << x <<
				// clearing a line from leftover floating point value
				 std::right << std::setw(precision + 3) << "\r";
			}
		}
	}
	std::cout << "Zakonczono";
	getchar();
}
