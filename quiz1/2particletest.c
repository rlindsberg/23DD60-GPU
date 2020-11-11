#include <stdio.h>

struct Position
{
    float x, y, z;
};

struct Particle
{
    struct Position position;
};

int main()
{
    struct Particle particle = { 0 };
    particle.position.x = 2;
    particle.position.y = 1;

}
