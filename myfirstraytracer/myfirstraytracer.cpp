// myfirstraytracer.cpp : Defines the entry point for the application.
//

#include "include/myfirstraytracer.h"

bool hit_anything(const ray& r, std::vector<std::shared_ptr<hitable>> world, double tmin, double tmax, intersection& isect) {
    bool hit = false;
    intersection temp_isect;
    double closest_t = tmax;
    for (int k = 0; k < world.size(); k++) {
        if (world[k]->hit(r, tmin, closest_t, temp_isect)) {
            hit = true;
            closest_t = temp_isect.hit_t;
            isect = temp_isect;
        }
    }
    return hit;
}

color traceray(ray r,std::vector<std::shared_ptr<hitable>> world) {
    intersection isect;

    if (hit_anything(r, world, 0.001, (double)FLT_MAX, isect)) {
        //return 0.5 * color(isect.hit_normal.x() + 1, isect.hit_normal.y() + 1, isect.hit_normal.z() + 1);
        return color(1, 0, 0);
    }
    else {
        vec3 unit_direction = r.direction.normalize();
        double a = 0.5 * (double(unit_direction.y()) + 1.0);
        //return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
        return color(0, 0, 0);
    }
}

int main() {
    std::vector<std::shared_ptr<hitable>> world;
    
    
    //add geometry 
    world.push_back(std::make_shared<sphere>(sphere(vec3(0,0,-1),0.5)));

    int n_channels = 3;
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 256;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

    uint8_t* pixels = new uint8_t[image_height*image_width*n_channels];
    double focal_length = 1.0;

    //place a camera at the origin (0,0,0)
    camera myCamera(vec3(0, 0, 0), 1.0, image_width,image_height,viewport_height);

    auto starttime = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            std::clog << "\rProgress: " << int(100 * (double(j * (image_width + 1)) + double(i)) / double(image_height * image_width)) << "%" << std::flush;

            //ray r(camera_center, pixel_center - camera_center);
            ray r = myCamera.getray(i, j);
        
            color pixelcolor = traceray(r,world);

            write_color((j*image_width + i)*n_channels, pixelcolor,pixels);

        }
    }
    std::clog << '\n';

    //get time for performance log
    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() / 1000000.0);

    std::clog << "Done! Duration: " << duration << "s" << "\n";

    stbi_write_png("C:/Users/oscar/source/repos/oscarbroddegard/raytracer/out/out.png",image_width,image_height,n_channels,pixels,image_width*n_channels);

    delete[] pixels; //phew

    return 0;
}