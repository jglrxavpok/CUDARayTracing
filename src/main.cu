#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include "cudautils.h"
#include <intersectables/Sphere.h>
#include <intersectables/IntersectableGroup.h>
#include "stb_image_write.h"
#include "rt.h"
#include "Camera.h"
#include "Material.h"
#include "materials/Lambertian.h"
#include "materials/Metal.h"
#include "materials/Dielectric.h"

using std::shared_ptr;
using std::vector;

static constexpr double ASPECT_RATIO = 16.0/9.0;
static constexpr int IMAGE_HEIGHT = 1080;
static constexpr int IMAGE_WIDTH = static_cast<int>(IMAGE_HEIGHT*ASPECT_RATIO);
static constexpr int MAX_BOUNCE = 20;

__device__ Color trace(const Ray& r, const Intersectable* world, curandState* rand, int remainingRays = MAX_BOUNCE) {
    Color skyBlue = Color(0.5, 0.7, 1.0);
    Color white = Color(1.0, 1.0, 1.0);
    Color red = Color(1.0, 0.0, 0.0);
    Color black = Color(0.0, 0.0, 0.0);

    if(remainingRays <= 0) {
        return black;
    }

    HitResult result{};
    if(world->hit(r, 0.001, INFINITY, result)) { // 0.001 to remove shadow acne
        Ray scattered{};
        Color attenuation{};
        if(result.material->scatter(r, result, rand, attenuation, scattered)) {
            return attenuation * trace(scattered, world, rand, remainingRays-1);
        }
        return white;
    }


    Vec3 direction = r.direction().normalized();
    // map from -1..1 to 0..1
    auto alpha = (direction.y() + 1.0) / 2.0;
    return (1.0-alpha) * white + alpha * skyBlue;
}

__global__
void rngInit(curandState* rngState) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
        return;
    int yInversed = IMAGE_HEIGHT-y-1;
    size_t ptr = (yInversed*IMAGE_WIDTH+x)*4;
    curand_init(1984, ptr, 0, &(rngState[ptr/4]));
}

__global__
void rayTrace(uint8_t* pixels, curandState* rngState, Intersectable** worldPtr) {
    const Intersectable* world = *worldPtr;
    Camera camera{Point3(-2,2,1), Point3(0,0,-1), Vec3(0,1,0), 20, ASPECT_RATIO};
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
        return;
    int yInversed = IMAGE_HEIGHT-y-1;
    size_t ptr = (yInversed*IMAGE_WIDTH+x)*4;

    curandState localRandState = rngState[ptr/4];
    Color pixelColor{};
    for(int i = 0; i < Camera::SAMPLES_PER_PIXEL; i++) {
        double dx = curand_uniform_double(&localRandState);
        double dy = curand_uniform_double(&localRandState);
        double u = double(x+dx) / (IMAGE_WIDTH-1);
        double v = double(y+dy) / (IMAGE_HEIGHT-1);

        pixelColor += trace(camera.generateRay(u, v), world, &localRandState);
    }

    writeColor(pixels, ptr, pixelColor, Camera::SAMPLES_PER_PIXEL);
}

Color traceHost(const Ray& r, const Intersectable& world, int remainingRays = MAX_BOUNCE) {
    Color skyBlue = Color(0.5, 0.7, 1.0);
    Color white = Color(1.0, 1.0, 1.0);
    Color red = Color(1.0, 0.0, 0.0);
    Color black = Color(0.0, 0.0, 0.0);

    if(remainingRays <= 0) {
        return black;
    }

    HitResult result{};
    if(world.hit(r, 0.001 /* remove shadow acne */, INFINITY, result)) {
        Ray scattered{};
        Color attenuation{};
        if(result.material->scatter(r, result, nullptr, attenuation, scattered)) {
            return attenuation * traceHost(scattered, world, remainingRays-1);
        }
        return white;
    }

    Vec3 direction = r.direction().normalized();
    // map from -1..1 to 0..1
    auto alpha = (direction.y() + 1.0) / 2.0;
    return (1.0-alpha) * white + alpha * skyBlue;
}

__global__
void worldInit(Intersectable** world, Intersectable** list) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto materialGround = new Lambertian(Color(0.8, 0.8, 0.0));
        auto materialCenter = new Lambertian(Color(0.1, 0.2, 0.5));
        auto materialRight = new Metal(Color(0.8, 0.6, 0.2), 0.0);
        auto materialLeft = new Dielectric(1.5);

        *list = new Sphere(Point3( 0.0, -100.5, -1.0), 100.0, materialGround);
        *(list+1) = new Sphere(Point3( 0.0,    0.0, -1.0), 0.5, materialCenter);
        *(list+2) = new Sphere(Point3( 1.0,    0.0, -1.0), 0.5, materialRight);
        *(list+3) = new Sphere(Point3( -1.0,    0.0, -1.0), 0.5, materialLeft);
        *(list+4) = new Sphere(Point3( -1.0,    0.0, -1.0), -0.45, materialLeft);
        *(world) = new IntersectableGroup(5, list);
    }
}

int main()
{
    uint8_t* pixels;
    checkCudaErrors(cudaMallocManaged(&pixels, sizeof(uint8_t)*IMAGE_WIDTH*IMAGE_HEIGHT*4));

    std::cout << "Allocated " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " image." << std::endl;

    int tileWidth = 8;
    int tileHeight = 8;
    dim3 blocks(IMAGE_WIDTH/tileWidth+1,IMAGE_HEIGHT/tileHeight+1);
    dim3 threads(tileWidth,tileHeight);

    curandState *deviceRNG;
    checkCudaErrors(cudaMalloc((void **)&deviceRNG, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(curandState)));

    rngInit<<<blocks, threads>>>(deviceRNG);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "RNG initialized" << std::endl;

    Intersectable** elements;
    Intersectable** world;
    checkCudaErrors(cudaMalloc(&elements, 5*sizeof(Intersectable*)));
    checkCudaErrors(cudaMalloc(&world, sizeof(Intersectable*)));

    worldInit<<<blocks, threads>>>(world, elements);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Scene initialized" << std::endl;

    rayTrace<<<blocks, threads>>>(pixels, deviceRNG, world);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Done, writing to file" << std::endl;

    stbi_write_png("./test.png", IMAGE_WIDTH, IMAGE_HEIGHT, 4, pixels, IMAGE_WIDTH*4);
    checkCudaErrors(cudaFree(pixels));

    // TODO: free
    return 0;
}