#include "simpleRT.h"

__device__ vec3 traceray(const ray& r,hitable* world) {
	intersection isect;
	if ((world)->hit(r, 0.001f, FLT_MAX, isect)) {
		return 0.5f * vec3(isect.hit_normal.x() + 1.0f, isect.hit_normal.y() + 1.0f, isect.hit_normal.z() + 1.0f);
	}
	else {
		vec3 unit_direction = unit_vector(r.getdirection());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}

__global__ void render(vec3* framebuffer, int width, int height,vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable* scenebuffer) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pidx = j * width + i;

	float u = float(i) / float(width);
	float v = float(j) / float(height);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical);

	framebuffer[pidx] = traceray(r,scenebuffer);
}

int main() {
	int imagewidth = 512;
	int imageheight = 512;
	int n_pixels = imageheight * imagewidth;

	vec3* framebuffer;
	checkCudaErrors(cudaMallocManaged((void**)&framebuffer, 3 * n_pixels * sizeof(vec3)));

	int threads_x = 8;
	int threads_y = 8;
	dim3 blocks(imagewidth / threads_x + 1, imageheight / threads_y + 1);
	dim3 threads(threads_x, threads_y);

	hitable* listbuffer;
	hitable* scenebuffer;
	listbuffer = new sphere(vec3(0, 0, -1), 0.5);
	listbuffer[1] = new sphere(vec3(0, -100.5, -1), 100);
	scenebuffer = new hitable_list(listbuffer, 2);
	cudaMallocManaged(&listbuffer, 2*sizeof(hitable*));
	cudaMallocManaged(&scenebuffer, sizeof(hitable*));




	render<<<blocks, threads >>>(framebuffer, imagewidth, imageheight,
								vec3(-2.0, -1.0, -1.0),
								vec3(4.0, 0.0, 0.0),
								vec3(0.0, 2.0, 0.0),
								vec3(0.0, 0.0, 0.0),
								scenebuffer);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());





	//pixel buffer to export to image
	uint8_t* pixels = new uint8_t[3 * n_pixels];

	// write to image
	for (int j = 0; j < imageheight; j++) {
		for (int i = 0; i < imagewidth; i++) {
			color pixelcolor = framebuffer[(j * imagewidth + i)];

			write_color((j * imagewidth + i) * 3, pixelcolor, pixels);

		}
	}

	stbi_write_png("C:/Users/oscar/source/repos/myfirstraytracer/out/out_simple.png", imagewidth, imageheight, 3, pixels, imagewidth * 3);

	delete[] pixels; //phew
}
