#include <stdio.h>

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
    Particle particle = { 0 };
    particle.position.x = 2;
    particle.position.y = 1;
}
