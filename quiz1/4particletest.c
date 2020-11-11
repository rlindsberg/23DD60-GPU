#include <stdio.h>
#include <stdlib.h>

#define NUM_PARTICLES 1024

typedef struct
{
    float x, y, z;
} Position;

typedef struct
{
   Position position;
} Particle;

int main()
{
    Particle *particles = (Particle *) malloc(NUM_PARTICLES * sizeof(Particle));

}
