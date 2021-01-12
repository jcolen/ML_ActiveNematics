#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#define PI 3.1415926535
#define PI2 1.570796327
#define MAX 65536
#define MAX2 32768
#define MAX34 49152

#define IDX(x, y, x_max) (y * x_max + x)
#define ANG2S(x) ((unsigned short)(x / PI * MAX))
#define S2ANG(x) (((double)x) / MAX * PI)
#define S2DEG(x) ((int)(((double)x) / MAX * 180))

unsigned short delta_theta(unsigned short * theta, int x, int y, int x_max)	{
	unsigned short th0, th1, dth;
	dth = 0;
	th0 = theta[IDX(x, y, x_max)];

	th1 = theta[IDX(x, (y+1), x_max)];
	dth += th1-th0;

	th1 = theta[IDX(x, (y-1), x_max)];
	dth += th1-th0;

	th1 = theta[IDX((x+1), y, x_max)];
	dth += th1-th0;
	
	th1 = theta[IDX((x-1), y, x_max)];
	dth += th1-th0;
	
	dth = (dth < MAX2) ? (dth / 4) : (MAX34 + dth / 4);
	return dth;
}

/**
Sharpen an input image using free energy minimization
Restricted now to 2d
Parameters:
	double * theta - Input theta field, with theta from 0 to pi
	double * coords
	int x_max, y_max
	int ncoords
	int fixed_border
	int nmax
*/
void sharpen(double * theta,
			 int * coords,
			 int ncoords,
			 int x_max, int y_max,
			 int fixed_border, int nmax)	{
	unsigned short *grid, *dgrid;
	bool * fixed;
	int x, y, x0, y0, xi, yi, n, m;

	grid = (unsigned short*) malloc(x_max * y_max * sizeof(short));
	dgrid = (unsigned short*) malloc(x_max * y_max * sizeof(short));
	fixed = (bool*) malloc(x_max * y_max * sizeof(bool));
	
	//Convert theta grid to [0, 65536) range
	for (x = 0; x < x_max * y_max; x ++)	{
		grid[x] = ANG2S(theta[x]);
		dgrid[x] = 0;
		fixed[x] = 0;
	}

	//Draw a box around each defect
	for (n = 0; n < ncoords; n ++)	{
		x0 = coords[2 * n];
		y0 = coords[2 * n + 1];
		printf("Defect at (%d,%d)\n", x0, y0);

		for (y = -fixed_border; y <= fixed_border; y ++)	{
			yi = y0 + y;
			if (yi < 0) yi = y_max + yi;
			if (yi >= y_max) yi -= y_max;
			for (x = -fixed_border; x <= fixed_border; x ++)	{
				xi = x0 + x;
				if (xi < 0)	xi = x_max + xi;
				if (xi >= x_max) xi -= x_max;
				fixed[IDX(xi, yi, x_max)] = 1;
			}
		}
	}
	//Fix the edge of the image
	for (y = 0; y < y_max; y ++)	{
		fixed[IDX(0, y, x_max)] = 1;
		fixed[IDX((x_max-1), y, x_max)] = 1;
	}
	for (x = 0; x < x_max; x ++)	{
		fixed[IDX(x, 0, x_max)] = 1;
		fixed[IDX(x, (y_max-1), x_max)] = 1;
	}

	//Iteratively adjust the grid
    for (n = 0; n < nmax; n++) {
		//Step 1. Solve for the region inside each box
		for (m = 0; m < ncoords; m ++)	{
			x0 = coords[2 * m];
			y0 = coords[2 * m + 1];
			for (y = -fixed_border+1; y < fixed_border; y ++)	{
				yi = y0 + y;
				if (yi < 0 || yi >= y_max)	{	continue;	}
				for (x = -fixed_border+1; x < fixed_border; x ++)	{
					xi = x0 + x;
					if (xi < 0 || xi >= x_max)	{	continue;	}
					dgrid[IDX(xi, yi, x_max)] = delta_theta(grid, xi, yi, x_max);
				}
			}
		}

        for (x = 0; x < x_max*y_max; x++) {
           	grid[x] += dgrid[x];
			grid[x] = grid[x] % MAX;
			dgrid[x] = 0;	//Zero the update field for the next iteration
        }
		//Step 2. Solve for the region outside each box
        for (y = 0; y < y_max; y ++)	{
			for (x = 0; x < x_max; x ++)	{
				if (fixed[IDX(x, y, x_max)])	{ continue;	}	
				//Gradient update outside the defect region
				dgrid[IDX(x, y, x_max)] = delta_theta(grid, x, y, x_max);
            }
        }

        for (x = 0; x < x_max*y_max; x++) {
            grid[x] += dgrid[x];
			grid[x] = grid[x] % MAX;
			dgrid[x] = 0;	//Zero the update field for the next iteration
        }
    }
    
	for (x = 0; x < x_max*y_max; x++) {
		theta[x] = S2ANG(grid[x]);
    }
    
	free(grid);
    free(dgrid);
	free(fixed);
}
