#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define PI 3.1415926535
#define PI2 1.570796327

#define IDX(x, y, x_max) (y * x_max + x)
#define DEG(x) ((int)(180 * x / PI))

/**
Compute the winding number around each pixel in an image
Restricted for now to 2d
*/
void winding(double * theta, int x_max, int y_max, int radius)	{
	double *dx, *dy;
	double dth;
	int x, y, i, xi, yi, idx;

	dx = (double*)malloc((x_max-1) * y_max * sizeof(double));
	dy = (double*)malloc(x_max * (y_max-1) * sizeof(double));
	
	//Gradient in x direction
	for (y = 0; y < y_max; y ++)	{
		for (x = 0; x < x_max-1; x ++)	{
			idx = IDX(x, y, x_max);
			dth = theta[idx + 1] - theta[idx];
			if (dth < -PI2)	dth += PI;
			else if (dth > PI2) dth -= PI;
			dx[IDX(x, y, (x_max-1))] = dth;
		}
	}
	//Gradient in y direction
	for (y = 0; y < y_max-1; y ++)	{
		for (x = 0; x < x_max; x ++)	{
			idx = IDX(x, y, x_max);
			dth = theta[idx + x_max] - theta[idx];
			if (dth < -PI2)	dth += PI;
			else if (dth > PI2) dth -= PI;
			dy[idx] = dth;
		}
	}
	
	//Compute winding about each point
	for (y = 0; y < y_max; y ++)	{
		for (x = 0; x < x_max; x ++)	{
			idx = IDX(x, y, x_max);
			theta[idx] = 0;
			if ( y < radius || y >= y_max - radius ||
				 x < radius || x >= x_max - radius)		{
				 continue;
			}
			for (i = -radius; i < radius; i ++)	{
				theta[idx] += dx[IDX((x+i), (y+radius), (x_max-1))];
				theta[idx] -= dx[IDX((x+i), (y-radius), (x_max-1))];
				theta[idx] -= dy[IDX((x+radius), (y+i), x_max)];
				theta[idx] += dy[IDX((x-radius), (y+i), x_max)];
			}
			theta[idx] /= 2 * PI;
		}
	}
	free(dx);
	free(dy);
}
