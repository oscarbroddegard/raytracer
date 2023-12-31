#pragma once

#ifndef COLOR_H
#define COLOR_H


#include <iostream>
#include <math.h>

#include "vec3.h"
using color = vec3;

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline void write_color(int index, color& c, uint8_t* pixels) {
    // Write the translated [0,255] value of each color component.
    
    for (int n = 0; n < 3; n++) {
        c.e[n] = pow(c.e[n],1.0/2.2);
        pixels[index + n] = (uint8_t)(256 * clamp(c.e[n], 0.0f, 0.999));
    }
}

#endif