// myfirstraytracer.cpp : Defines the entry point for the application.
//

#include "include/myfirstraytracer.h"

color traceray(ray r) {
    sphere center_sphere(vec3(0,0,-1),0.5);
    if (center_sphere.hit(r)) { return color(1, 0, 0); }
    vec3 unit_direction = r.direction.normalize();
    double a = 0.5 * (double(unit_direction.y()) + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

int main() {
    std::string outputfile = "out/output.txt";
    

    std::ofstream out(outputfile);
    int n_channels = 3;
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 256;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

    uint8_t* pixels = new uint8_t[image_height*image_width*n_channels];

    vec3 viewport_u(viewport_width, 0, 0);
    vec3 viewport_v(0, -viewport_height, 0);

    vec3 pdeltax = viewport_u / image_width;
    vec3 pdeltay = viewport_v / image_height;

    double focal_length = 1.0;
    vec3 camera_center(0, 0, 0);

    vec3 view_upperleft = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    vec3 pixel00_location = view_upperleft + 0.5 * (pdeltax + pdeltay);

    out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            std::clog << "\rProgress: " << int(100 * (double(j * (image_width + 1)) + double(i)) / double(image_height * image_width)) << "%" << std::flush;
            vec3 pixel_center = pixel00_location + j * pdeltay + i * pdeltax;

            ray r(camera_center, pixel_center - camera_center);

            color pixelcolor = traceray(r);

            write_color((j*image_width + i)*n_channels, pixelcolor,pixels);

        }
    }
    std::clog << '\n';

    stbi_write_png("out.png",image_width,image_height,n_channels,pixels,image_width*n_channels);

    delete[] pixels; //phew

    return 0;
}