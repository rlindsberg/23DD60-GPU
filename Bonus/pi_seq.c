
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    int count = 0;
    double x, y, z, pi, a, b;

    srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!

    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < NUM_ITER; iter++)
    {
        // Generate random (X,Y) points
        a = (double)random();
        b = (double)RAND_MAX;
        x = a / b;
        printf("The first is %f\n", a);
        printf("The second is %f\n", b);
        printf("The res is %f\n", x);

        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));

        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }

    // Estimate Pi and display the result
    pi = ((double)count / (double)NUM_ITER) * 4.0;

    printf("The result is %f\n", pi);

    return 0;
}
