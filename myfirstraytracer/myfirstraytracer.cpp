#include "include/myfirstraytracer.h"

bool hit_anything(const ray& r, std::vector<std::shared_ptr<hitable>> hitables, double tmin, double tmax, intersection& isect) {
    bool hit = false;
    intersection temp_isect;
    double closest_t = tmax;
    for (int k = 0; k < hitables.size(); k++) {
        if (hitables[k]->hit(r, tmin, closest_t, temp_isect)) {
            hit = true;
            closest_t = temp_isect.hit_t;
            isect = temp_isect;
        }
    }
    return hit;
}

vec3 random_in_unit_sphere() {
    vec3 p;
    do {
        p = 2.0*vec3((double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX)-vec3(1,1,1);
    } while (p.norm() >= 1.0);
    return p;
}



color traceray(ray r,scene world,int depth) {
    intersection isect,shadow;

    if (hit_anything(r, world.hitables, 0.001, FLT_MAX, isect)) {
        color pixelcolor = isect.hit_material.diffuse_color;

        //shadowing and lighting
        double diff = 0.0;
        double shadowed = 0.0;
        for (int i = 0; i < world.light_sources.size(); i++) {
            vec3 L = world.light_sources[i] - isect.hit_position;
            ray shadowray = getshadowray(isect.hit_position,world.light_sources[i]);
            if (hit_anything(shadowray, world.hitables, 0.001, L.norm(), shadow)) {
                shadowed += shadow.hit_material.transparency > 0 ? 1.0-shadow.hit_material.transparency : 1.0;
            }
            diff += dot(L, isect.hit_normal) > 0 ? dot(L, isect.hit_normal) : 0;
        }
        pixelcolor *= shadowed > 0 ? diff/shadowed : diff;

        //reflection
        color refcolor;
        if (isect.hit_material.reflectivity > 0 && depth > 0) {
            ray reflectedray = getreflectedray(r.direction, isect.hit_normal, isect.hit_position);
            refcolor = traceray(reflectedray, world,depth-1);
        }

        //refraction
        color refractedcolor;
        if (isect.hit_material.transparency > 0 && depth > 0) {
            ray refractedray = getrefractedray(r.direction, isect.hit_normal, isect.hit_position, isect.hit_material.refractive_index);
            refractedcolor = traceray(refractedray, world, depth - 1);
        }

        pixelcolor = (1 - isect.hit_material.reflectivity - isect.hit_material.transparency) * pixelcolor + isect.hit_material.reflectivity * refcolor + isect.hit_material.transparency * refractedcolor;
        

        return 0.5*pixelcolor;
    }
    else { //background
        vec3 unit_direction = r.direction.normalize(); 
        double a = 0.5 * (double(unit_direction.y()) + 1.0);
        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
    }
}

int main() {
    
    material red_reflective(color(0.7, 0.1, 0.0).normalize(), 0.5, 0.1,1.0);
    material white_reflective(color(1, 1, 1).normalize(), 0.3, 0.5,1.3);

    scene world;
    
    //add geometry 
    world.add_hitable(std::make_shared<sphere>(sphere(vec3(0,0,-1),0.5,red_reflective)));
    world.add_hitable(std::make_shared<sphere>(sphere(vec3(0, -100.5, -1), 100,white_reflective)));

    world.add_light(vec3(-2.0, 2.0, -1.0));
    //world.add_light(vec3(2.0, 1.0, -1.0));
    

    int n_channels = 3;
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 256;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    auto viewport_height = 2.0;

    uint8_t* pixels = new uint8_t[image_height*image_width*n_channels];
    double focal_length = 1.0;

    //place a camera at the origin (0,0,0)
    camera myCamera(vec3(0, 0, 3), 1.0, image_width,image_height,viewport_height);

    auto starttime = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            std::clog << "\rProgress: " << int(100 * (double(j * (image_width + 1)) + double(i)) / double(image_height * image_width)) << "%" << std::flush;

            //ray r(camera_center, pixel_center - camera_center);
            ray r = myCamera.getray(i, j);
        
            color pixelcolor = traceray(r,world,4);

            write_color((j*image_width + i)*n_channels, pixelcolor,pixels);

        }
    }
    std::clog << '\n';

    //get time for performance log
    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() / 1000000.0);

    std::clog << "Done! Duration: " << duration << "s" << "\n";

    stbi_write_png("C:/Users/oscar/source/repos/myfirstraytracer/out/out.png",image_width,image_height,n_channels,pixels,image_width*n_channels);

    delete[] pixels; //phew

    return 0;
}