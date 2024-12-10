#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#define PI 3.1415926535
#define PI2 1.570796327

#define IDX(x, y, x_max) (y * x_max + x)
#define DEG(x) ((int)(180 * x / PI))

double delta_theta(double * theta, int x, int y, int x_max)	{
	double th0, th1, dth;
	dth = 0;
	th0 = theta[IDX(x, y, x_max)];

	th1 = theta[IDX(x, (y+1), x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	th1 = theta[IDX(x, (y-1), x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	th1 = theta[IDX((x+1), y, x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	th1 = theta[IDX((x-1), y, x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	dth /= 4;
	return dth;
}

/**
Sharpen an input image arrays using free energy minimization
Restricted to now to 2d
Parameters:
	double * in - flattened input image to be sharpened (input = sin(2theta)
	double * coords - flattened array of coordinates of topological defects
	int x_max, y_max - Dimensions of the array
	int ncoords - Number of coordinates
	int fixed_border - Size of border to draw around each defect
*/
void sharpen(double * th, 
			 int * coords, 
			 int ncoords, 
			 int x_max, int y_max, 
			 int fixed_border,
			 int nmax)	{
    double *dth;
	bool * fixed;
	int x, y, x0, y0, xi, yi, n, m;

    dth= (double*)malloc( x_max*y_max*sizeof(double));
	fixed = (bool*)malloc(x_max*y_max*sizeof(bool));

	//Initialize each array
	for (x = 0; x < x_max * y_max; x++) {
		fixed[x] = 0;
		dth[x] = 0;
	}

	//Draw a box around each defect
	for (n = 0; n < ncoords; n ++)	{
		x0 = coords[2 * n];
		y0 = coords[2 * n + 1];
		//printf("Defect at (%d,%d)\n", x0, y0);

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

	//Iteratively adjust the theta field
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
					dth[IDX(xi, yi, x_max)] = delta_theta(th, xi, yi, x_max);
				}
			}
		}

        for (x = 0; x < x_max*y_max; x++) {
            th[x] += dth[x];
			th[x] = fmod(th[x], PI);
			dth[x] = 0;	//Zero the update field for the next iteration
        }
		//Step 2. Solve for the region outside each box
        for (y = 0; y < y_max; y ++)	{
			for (x = 0; x < x_max; x ++)	{
				if (fixed[IDX(x, y, x_max)])	{ continue;	}	
				//Gradient update outside the defect region
				dth[IDX(x, y, x_max)] = delta_theta(th, x, y, x_max);
            }
        }

        for (x = 0; x < x_max*y_max; x++) {
            th[x] += dth[x];
			th[x] = fmod(th[x], PI);
			dth[x] = 0;	//Zero the update field for the next iteration
        }
    }
    
    free(dth);
	free(fixed);
}

double delta_theta_periodic(double * theta, int x, int y, int x_max, int y_max)	{
	double th0, th1, dth;
	dth = 0;
	th0 = theta[IDX(x, y, x_max)];

	if (y + 1 < y_max)	th1 = theta[IDX(x, (y+1), x_max)] - th0;
	else				th1 = theta[IDX(x, (y+1-y_max), x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	if (y > 0)	th1 = theta[IDX(x, (y-1), x_max)] - th0;
	else		th1 = theta[IDX(x, (y-1+y_max), x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	if (x + 1 < y_max)	th1 = theta[IDX((x+1), y, x_max)] - th0;
	else				th1 = theta[IDX((x+1-x_max), y, x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	if (x > 0)	th1 = theta[IDX((x-1), y, x_max)] - th0;
	else		th1 = theta[IDX((x-1+x_max), y, x_max)] - th0;
	if (th1 < -PI2) th1 += PI;
	else if (th1 > PI2) th1 -= PI;
	dth += th1;
	
	dth /= 4;
	return dth;
}

/**
Sharpening with periodic boundary conditions
*/
void sharpen_periodic(double * th, 
			 		  int * coords, 
			 		  int ncoords, 
			 		  int x_max, int y_max, 
			 		  int fixed_border,
			 		  int nmax)	{
    double *dth;
	bool * fixed;
	int x, y, x0, y0, xi, yi, n, m;

    dth= (double*)malloc( x_max*y_max*sizeof(double));
	fixed = (bool*)malloc(x_max*y_max*sizeof(bool));

	//Initialize each array
	for (x = 0; x < x_max * y_max; x++) {
		fixed[x] = 0;
		dth[x] = 0;
	}

	//Draw a box around each defect
	for (n = 0; n < ncoords; n ++)	{
		x0 = coords[2 * n];
		y0 = coords[2 * n + 1];
		//printf("Defect at (%d,%d)\n", x0, y0);

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
	
	//Iteratively adjust the theta field
    for (n = 0; n < nmax; n++) {
		//Step 1. Solve for the region inside each box
		for (m = 0; m < ncoords; m ++)	{
			x0 = coords[2 * m];
			y0 = coords[2 * m + 1];
			for (y = -fixed_border+1; y < fixed_border; y ++)	{
				yi = y0 + y;
				if (yi < 0) yi = y_max + yi;
				if (yi >= y_max) yi -= y_max;
				for (x = -fixed_border+1; x < fixed_border; x ++)	{
					xi = x0 + x;
					if (xi < 0) xi = x_max + xi;
					if (xi >= x_max) xi -= x_max;
					dth[IDX(xi, yi, x_max)] = delta_theta_periodic(th, xi, yi, x_max, y_max);
				}
			}
		}

        for (x = 0; x < x_max*y_max; x++) {
            th[x] += dth[x];
			th[x] = fmod(th[x], PI);
			dth[x] = 0;	//Zero the update field for the next iteration
        }
		//Step 2. Solve for the region outside each box
        for (y = 0; y < y_max; y ++)	{
			for (x = 0; x < x_max; x ++)	{
				if (fixed[IDX(x, y, x_max)])	{ continue;	}	
				//Gradient update outside the defect region
				dth[IDX(x, y, x_max)] = delta_theta_periodic(th, x, y, x_max, y_max);
            }
        }

        for (x = 0; x < x_max*y_max; x++) {
            th[x] += dth[x];
			th[x] = fmod(th[x], PI);
			dth[x] = 0;	//Zero the update field for the next iteration
        }
    }
    
    free(dth);
	free(fixed);
}
