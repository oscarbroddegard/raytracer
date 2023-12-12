#include "raytracer_cuda.h"

__device__ bool sphere_hit(const ray& r,sphere& target, float tmin, float tmax, intersection& isect) {
	vec3 OC = r.origin - target.center;
	float a = dot(r.direction, r.direction);
	float b = dot(OC, r.direction);
	float c = dot(OC, OC) - target.radius * target.radius;

	float disc = b * b - a * c;
	if (disc > 0) {
		float temp = (-b - sqrt(disc)) / a;
		if (temp < tmax && temp > tmin) {
			isect.hit_t = temp;
			isect.hit_position = r.at(isect.hit_t);
			isect.hit_normal = (isect.hit_position - target.center) / target.radius;
			isect.hit_material = target.sphere_material;
			return true;
		}
		temp = (-b + sqrt(disc)) / a;
		if (temp < tmax && temp > tmin) {
			isect.hit_t = temp;
			isect.hit_position = r.at(isect.hit_t);
			isect.hit_normal = ((isect.hit_position - target.center) / target.radius).normalize();
			isect.hit_material = target.sphere_material;
			return true;
		}
	}

	return false;
}

__device__ vec3 traceray(ray r, sphere_list* world, int depth) {
    intersection isect, shadow;

    if ((world)->hit(r, 0.01, FLT_MAX, isect)) {
        vec3 pixelcolor = isect.hit_material->diffuse_color;

        //shadowing and lighting
        float diff = 0.0;
        vec3 light_source(0.0, 30.0, -2.0);
        vec3 L = light_source - isect.hit_position;
        ray shadowray = getshadowray(isect.hit_position, light_source);
        if ((world)->hit(r, 0.01, L.norm(), shadow)) {
            diff = 0.0;
        }
        else {
            diff += dot(L.normalize(), isect.hit_normal) > 0 ? dot(L.normalize(), isect.hit_normal) : 0;
        }
        pixelcolor *= diff;

        //reflection
        vec3 refcolor;
        if (isect.hit_material->reflectivity > 0 && depth > 0) {
            ray reflectedray = getreflectedray(r.direction.normalize(), isect.hit_normal, isect.hit_position);
            refcolor = traceray(reflectedray, world, depth - 1);
        }

        //refraction
        vec3 refractedcolor;
        if (isect.hit_material->transparency > 0 && depth > 0) {
            ray refractedray = getrefractedray(r.direction.normalize(), isect.hit_normal, isect.hit_position, isect.hit_material->refractive_index);
            refractedcolor = traceray(refractedray, world, depth - 1);
        }

        pixelcolor = (1 - isect.hit_material->transparency - isect.hit_material->reflectivity) * pixelcolor + isect.hit_material->reflectivity * refcolor + isect.hit_material->transparency * refractedcolor;


        return 0.5 * pixelcolor;
    }
    else { //background
        return vec3(0.0, 0.0, 0.0);
    }
}

__global__ void render(sphere_list* scenebuffer, vec3* framebuffer, camera* CUDAcam, int image_width, int image_height,int depth = 4) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height)) return;
    ray r = (CUDAcam)->getray(((float)i) + 0.5, ((float)j) + 0.5);
    int pidx = j * image_width + i;
    framebuffer[pidx] = traceray(r, scenebuffer, depth);
}

__global__ void bindSceneBuffer(sphere* hitlist, sphere_list* buffer_ptr, camera* camerabuffer) {
    vec3 eye(0.0f, 10.0f, 30.0f);
    vec3 lookAt(0.0f, 10.0f, -5.0f);
    vec3 up(0.0f, 1.0f, 0.0f);
    camerabuffer = new camera(eye, lookAt, up, 52.0f, 1.0f, 512, 512);

    material whiteDiffuse = material(color(0.9f, 0.9f, 0.9f), 0.0f, 0.0f, 1.0f);
    material greenDiffuse = material(color(0.1f, 0.6f, 0.1f), 0.0f, 0.0f, 1.0f);
    material redDiffuse = material(color(1.0f, 0.1f, 0.1f), 0.0f, 0.0f, 1.0f);
    material blueDiffuse = material(color(0.0f, 0.2f, 0.9f), 0.0f, 0.0f, 1.0f);
    material yellowReflective = material(color(1.0f, 0.6f, 0.1f), 0.2f, 0.0f, 1.0f);
    material transparent = material(color(1.0f, 1.0f, 1.0f), 0.2f, 0.8f, 1.3f);

    //add geometry 
    hitlist[0] = sphere(vec3(0, 3, -20), 3.0f, new material(color(1.0f, 0.1f, 0.1f), 0.0f, 0.0f, 1.0f));
    hitlist[1] = sphere(vec3(-7.0f, 3.0f, -20.0f), 3.0f, new material(color(0.0f, 0.2f, 0.9f), 0.0f, 0.0f, 1.0f));
    hitlist[2] = sphere(vec3(7.0f, 3.0f, -20.0f), 3.0f, new material(color(0.1f, 0.6f, 0.1f), 0.0f, 0.0f, 1.0f));
    buffer_ptr = new sphere_list(hitlist, 22 * 22 + 1 + 3);
    /*world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, 50.0), vec3(20.0, 0.0, 50.0), vec3(20.0, 0.0, -50.0), whiteDiffuse))); //floor
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, 50.0), vec3(20.0, 0.0, -50.0), vec3(-20.0, 0.0, -50.0), whiteDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, -50.0), vec3(20.0, 0.0, -50.0), vec3(20.0, 40.0, -50.0), whiteDiffuse))); //back wall
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, -50.0), vec3(20.0, 40.0, -50.0), vec3(-20.0, 40.0, -50.0), whiteDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 40.0, 50.0), vec3(-20.0, 40.0, -50.0), vec3(20.0, 40.0, 50.0), whiteDiffuse))); //ceiling
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(20.0, 40.0, 50.0), vec3(-20.0, 40.0, -50.0), vec3(20.0, 40.0, -50.0), whiteDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, 50.0), vec3(-20.0, 40.0, -50.0), vec3(-20.0, 40.0, 50.0), redDiffuse))); // red wall
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(-20.0, 0.0, 50.0), vec3(-20.0, 0.0, -50.0), vec3(-20.0, 40.0, -50.0), redDiffuse)));

    world.add_hitable(std::make_shared<triangle>(triangle(vec3(20.0, 0.0, 50.0), vec3(20.0, 40.0, -50.0), vec3(20.0, 40.0, 50.0), greenDiffuse))); // green wall
    world.add_hitable(std::make_shared<triangle>(triangle(vec3(20.0, 0.0, 50.0), vec3(20.0, 0.0, -50.0), vec3(20.0, 40.0, -50.0), greenDiffuse))); */

    //world.add_hitable(std::make_shared<sphere>(sphere(vec3(7.0, 3.0, 0.0), 3.0, yellowReflective)));
    //world.add_hitable(std::make_shared<sphere>(sphere(vec3(9.0, 10.0, 0.0), 3.0, yellowReflective)));

    //world.add_hitable(std::make_shared<sphere>(sphere(vec3(-7.0, 3.0, 0.0), 3.0, transparent)));
    //world.add_hitable(std::make_shared<sphere>(sphere(vec3(-9.0, 10.0, 0.0), 3.0, transparent)));

    //world.add_light(vec3(0.0, 30.0, -2.0));
    //world.add_light(vec3(2.0, 1.0, -1.0));
}

int main() {

    

    //scene world;

    //createScene(world);

    


    int n_channels = 3;
    auto aspect_ratio = 1.0;
    int image_width = 512;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    int n_pixels = image_height * image_width;

    //set up framebuffer
    vec3* framebuffer;
    size_t fb_size = sizeof(vec3) * n_pixels;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, fb_size));

    //set up thread block sizes
    int threads_x = 8;
    int threads_y = 8;
    dim3 blocks(image_width / threads_x + 1, image_height / threads_y + 1);
    dim3 threads(threads_x, threads_y);

    //place a camera at the origin (0,0,0)
    
    camera* camerabuffer;
    checkCudaErrors(cudaMalloc((void**) &camerabuffer, sizeof(camera)));
    
    //set up scenebuffer
    sphere* list_of_hitables;
    int n_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**) &list_of_hitables, n_hitables * sizeof(sphere)));
    sphere_list* scenebuffer;
    checkCudaErrors(cudaMalloc((void**)&scenebuffer, sizeof(sphere_list)));

    bindSceneBuffer<<<1, 1>>>(list_of_hitables, scenebuffer,camerabuffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    


    std::clog << "Starting render" << std::endl;
    auto starttime = std::chrono::high_resolution_clock::now();
    //send everything to GPU
    render<<<blocks, threads >>>(scenebuffer,framebuffer,camerabuffer,image_width,image_height);
    //failsafe
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::clog << std::flush;

    //get time for performance log
    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = (std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() / 1000000.0);

    std::clog << "Done! Duration: " << duration << "s" << "\n";

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
    freeBuffers<<<1, 1 >>>(list_of_hitables, scenebuffer, n_hitables,camerabuffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(scenebuffer));
    checkCudaErrors(cudaFree(framebuffer));

    return 0;
}