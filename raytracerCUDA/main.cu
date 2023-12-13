#include "raytracer_cuda.h"

__device__ bool hit_list(const ray& r, sphere_list* world, float tmin, float tmax, intersection& isect) {
    bool hit = false;
    intersection temp_isect;
    double closest_t = tmax;
    for (int k = 0; k < world->n_hitables; k++) {
        if (sphere_hit(r, world->list[k], tmin, closest_t, temp_isect)) {
            hit = true;
            closest_t = temp_isect.hit_t;
            isect = temp_isect;
        }
    }
    return hit;
}

__device__ vec3 traceray(ray r, sphere_list* world, int depth) {
    intersection isect, shadow;
    vec3 unit_direction = r.getdirection(); //keep

    float t = 0.5f * (unit_direction.normalize().y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
    
    if (hit_list(r,world, 0.01, FLT_MAX, isect)) {
        vec3 pixelcolor = isect.hit_material.diffuse_color;

        //shadowing and lighting
        float diff = 0.0;
        vec3 light_source(0.0, 30.0, -2.0);
        vec3 L = light_source - isect.hit_position;
        ray shadowray = getshadowray(isect.hit_position, light_source);
        if (hit_list(r, world, 0.01, L.norm(), shadow)) {
            diff = 0.0;
        }
        else {
            diff += dot(L.normalize(), isect.hit_normal) > 0 ? dot(L.normalize(), isect.hit_normal) : 0;
        }
        pixelcolor *= diff;

        //reflection
        vec3 refcolor;
        if (isect.hit_material.reflectivity > 0 && depth > 0) {
            ray reflectedray = getreflectedray(unit_direction.normalize(), isect.hit_normal, isect.hit_position);
            refcolor = traceray(reflectedray, world, depth - 1);
        }

        //refraction
        vec3 refractedcolor;
        if (isect.hit_material.transparency > 0 && depth > 0) {
            ray refractedray = getrefractedray(unit_direction.normalize(), isect.hit_normal, isect.hit_position, isect.hit_material.refractive_index);
            refractedcolor = traceray(refractedray, world, depth - 1);
        }

        pixelcolor = (1 - isect.hit_material.transparency - isect.hit_material.reflectivity) * pixelcolor + isect.hit_material.reflectivity * refcolor + isect.hit_material.transparency * refractedcolor;


        return pixelcolor;
    }
    else { //background
        vec3 unit_direction = r.getdirection();
        float t = 0.5f * (unit_direction.normalize().y() + 1.0f);
        return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
    }
}

__global__ void render(sphere_list* scenebuffer, vec3* framebuffer, camera* CUDAcam, int image_width, int image_height,int depth = 4) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height)) return;
    ray r = CUDAcam->getray(((float)i) + 0.5, ((float)j) + 0.5);
    int pidx = j * image_width + i;
    framebuffer[pidx] = traceray(r, scenebuffer, depth);
    //framebuffer[pidx] = vec3((float)i / image_width, CUDAcam->getaspect(), 0.0f);
}

int main() {

    int n_channels = 3;
    auto aspect_ratio = 1.0;
    int image_width = 512;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    int n_pixels = image_height * image_width;

    //set up framebuffer
    vec3* framebuffer;
    size_t fb_size = sizeof(vec3) * n_pixels;
    checkCudaErrors(cudaMallocManaged(&framebuffer, fb_size));

    //set up thread block sizes
    int threads_x = 16;
    int threads_y = 16;
    dim3 blocks(image_width / threads_x + 1, image_height / threads_y + 1);
    dim3 threads(threads_x, threads_y);


    //place a camera
    vec3 eye(0.0f, 10.0f, 30.0f);
    vec3 lookAt(0.0f, 10.0f, -5.0f);
    vec3 up(0.0f, 1.0f, 0.0f);
    camera CUDAcam(vec3(0.0f, 10.0f, 30.0f), vec3(0.0f, 10.0f, -5.0f), vec3(0.0f, 1.0f, 0.0f), 52.0f, 1.0f, 512, 512);
    //std::clog << sizeof(CUDAcam);
    camera* camerabuffer;
    checkCudaErrors(cudaMalloc(&camerabuffer, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(camerabuffer, &CUDAcam, sizeof(camera), cudaMemcpyHostToDevice));

    //set up scenebuffer
    std::vector<sphere> scene;
    scene.push_back(sphere(vec3(0, 3, -20), 3.0f, material(color(1.0f, 0.1f, 0.1f), 0.0f, 0.0f, 1.0f)));
    scene.push_back(sphere(vec3(-7.0f, 3.0f, -20.0f), 3.0f, material(color(0.0f, 0.2f, 0.9f), 0.0f, 0.0f, 1.0f)));
    scene.push_back(sphere(vec3(7.0f, 3.0f, -20.0f), 3.0f, material(color(0.1f, 0.6f, 0.1f), 0.0f, 0.0f, 1.0f)));
    int n_hitables = static_cast<int>(scene.size());
    //scene[0] = sphere(vec3(0, 3, -20), 3.0f, material(color(1.0f, 0.1f, 0.1f), 0.0f, 0.0f, 1.0f));
    //scene[1] = sphere(vec3(-7.0f, 3.0f, -20.0f), 3.0f, material(color(0.0f, 0.2f, 0.9f), 0.0f, 0.0f, 1.0f));
    //scene[2] = sphere(vec3(7.0f, 3.0f, -20.0f), 3.0f, material(color(0.1f, 0.6f, 0.1f), 0.0f, 0.0f, 1.0f));

    sphere_list* scenebuffer;
    sphere_list list = sphere_list(scene.data(), n_hitables);
    std::clog << "checking" << "\n";
    checkCudaErrors(cudaMalloc(&scenebuffer, sizeof(scenebuffer)));
    //std::clog << "checking" << "\n";
    //checkCudaErrors(cudaMalloc(&(scenebuffer.list), n_hitables * sizeof(sphere)));
    //std::clog << "checking" << "\n";
    //checkCudaErrors(cudaMemcpy(scenebuffer.list, scene.data(), n_hitables * sizeof(sphere), cudaMemcpyHostToDevice));
    std::clog << "checking" << "\n";
    checkCudaErrors(cudaMemcpy(scenebuffer, &list, sizeof(sphere_list), cudaMemcpyHostToDevice));



    std::clog << "Starting render" << std::endl;
    auto starttime = std::chrono::high_resolution_clock::now();
    //send everything to GPU
    render<<<blocks, threads >>>(scenebuffer,framebuffer,camerabuffer,image_width,image_height);
    //get time for performance log
    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() / 1000000.0);

    std::clog << "Done! Duration: " << duration << "s" << "\n";

    //failsafe
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    

    //pixel buffer to export to image
    uint8_t* pixels = new uint8_t[n_channels*n_pixels];
    
    // write to image
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            color pixelcolor = framebuffer[(j * image_width + i)];

            write_color((j * image_width + i) * n_channels, pixelcolor, pixels);

        }
    }

    stbi_write_png("C:/Users/oscar/source/repos/myfirstraytracer/out/out_cuda.png", image_width, image_height, n_channels, pixels, image_width * n_channels);

    delete[] pixels; //phew

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(camerabuffer));
    checkCudaErrors(cudaFree(scenebuffer));
    checkCudaErrors(cudaFree(framebuffer));

    return 0;
}