#pragma once

#ifndef COLOR_H
#define COLOR_H


#include <iostream>

#include "vec3.h"
using color = vec3;

void write_color(std::ostream &out,const color& c) {
    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(255.999 * c.e[0]) << ' '
        << static_cast<int>(255.999 * c.e[1]) << ' '
        << static_cast<int>(255.999 * c.e[2]) << '\n';
}

#endif