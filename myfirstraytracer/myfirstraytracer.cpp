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

    if (hit_anything(r, world.hitables, 0.01, FLT_MAX, isect)) {
        color pixelcolor = isect.hit_material.diffuse_color;

        //shadowing and lighting
        double diff = 0.0;
        double shadowed = 0.0;
        for (int i = 0; i < world.light_sources.size(); i++) {
            vec3 L = world.light_sources[i] - isect.hit_position;
            ray shadowray = getshadowray(isect.hit_position,world.light_sources[i]);
            if (hit_anything(shadowray, world.hitables, 0.01, L.norm(), shadow)) {
                diff = 0.0;
                break;
            }
            diff += dot(L.normalize(), isect.hit_normal) > 0 ? dot(L, isect.hit_normal) : 0;
        }
        pixelcolor *= diff;

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

        pixelcolor = (1-isect.hit_material.transparency-isect.hit_material.reflectivity)*pixelcolor + isect.hit_material.reflectivity * refcolor + isect.hit_material.transparency * refractedcolor;
        

        return 0.5*pixelcolor;
    }
    else { //background
        return color(0.0, 0.0, 0.0);
    }
}

int main() {
    
    material whiteDiffuse = material(color(0.9f, 0.9f, 0.9f), 0.0f, 0.0f, 1.0f);
    material greenDiffuse = material(color(0.1f, 0.6f, 0.1f), 0.0f, 0.0f, 1.0f);
    material redDiffuse = material(color(1.0f, 0.1f, 0.1f), 0.0f, 0.0f, 1.0f);
    material blueDiffuse = material(color(0.0f, 0.2f, 0.9f), 0.0f, 0.0f, 1.0f);
    material yellowReflective = material(color(1.0f, 0.6f, 0.1f), 0.2f, 0.0f, 1.0f);
    material transparent = material(color(1.0f, 1.0f, 1.0f), 0.2f, 0.8f, 1.3f);

    scene world;
    
    //add geometry 
    world.add_hitable(std::make_shared<sphere>(sphere(vec3(0, 3, -20), 3.0, redDiffuse)));
    world.add_hitable(std::make_shared<sphere>(sphere(vec3(-7.0, 3.0, -20.0), 3.0, blueDiffuse)));
    world.add_hitable(std::make_shared<sphere>(sphere(vec3(7.0, 3.0, -20.0), 3.0, greenDiffuse)));
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0,0.0,50.0),vec3(20.0,0.0,50.0),vec3(20.0,0.0,-50.0),whiteDiffuse))); //floor
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, 50.0), vec3(20.0, 0.0, -50.0), vec3(-20.0, 0.0, -50.0), whiteDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, -50.0), vec3(20.0, 0.0, -50.0), vec3(20.0, 40.0, -50.0), whiteDiffuse))); //back wall
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, -50.0), vec3(20.0, 40.0, -50.0), vec3(-20.0, 40.0, -50.0), whiteDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 40.0, 50.0), vec3(-20.0, 40.0, -50.0), vec3(20.0, 40.0, 50.0), whiteDiffuse))); //ceiling
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(20.0, 40.0, 50.0), vec3(-20.0, 40.0, -50.0), vec3(20.0, 40.0, -50.0), whiteDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, 50.0), vec3(-20.0, 40.0, -50.0), vec3(-20.0, 40.0, 50.0), redDiffuse))); // red wall
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, 50.0), vec3(-20.0, 0.0, -50.0), vec3(-20.0, 40.0, -50.0), redDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(20.0, 0.0, 50.0), vec3(20.0, 40.0, -50.0), vec3(20.0, 40.0, 50.0), greenDiffuse))); // green wall
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(20.0, 0.0, 50.0), vec3(20.0, 0.0, -50.0), vec3(20.0, 40.0, -50.0), greenDiffuse)));

    world.add_hitable(std::make_shared<sphere>(sphere(vec3(7.0, 3.0, 0.0), 3.0, yellowReflective)));
    world.add_hitable(std::make_shared<sphere>(sphere(vec3(9.0, 10.0, 0.0), 3.0, yellowReflective)));

    //world.add_hitable(std::make_shared<sphere>(sphere(vec3(-7.0, 3.0, 0.0), 3.0, transparent)));
    //world.add_hitable(std::make_shared<sphere>(sphere(vec3(-9.0, 10.0, 0.0), 3.0, transparent)));

    world.add_light(vec3(0.0, 30.0, -5.0));
    //world.add_light(vec3(2.0, 1.0, -1.0));
    

    int n_channels = 3;
    auto aspect_ratio = 1.0;
    int image_width = 256;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    uint8_t* pixels = new uint8_t[image_height*image_width*n_channels];
    double focal_length = 1.0;

    //place a camera at the origin (0,0,0)
    vec3 eye(0.0f, 10.0f, 30.0f);
    vec3 lookAt(0.0f, 10.0f, -5.0f);
    vec3 up(0.0f, 1.0f, 0.0f);
    camera myCamera(eye, lookAt, up, 52.0, aspect_ratio,image_width,image_height);
    const int depth = 4;

    auto starttime = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            std::clog << "\rProgress: " << int(100 * (double(j * (image_width + 1)) + double(i)) / double(image_height * image_width)) << "%" << std::flush;

            //ray r(camera_center, pixel_center - camera_center);
            ray r = myCamera.getray(((double)i)+0.5, ((double)j) + 0.5);
        
            color pixelcolor = traceray(r,world,depth);

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