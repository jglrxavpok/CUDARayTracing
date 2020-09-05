//
// Created by jglrxavpok on 05/09/2020.
//

#include "Texture.h"
#include "stb_image.h"
#include "cudautils.h"

__device__ Texture::Texture(int width, int height, Color *pixels): width(width), height(height), pixels(pixels) {}

__device__ Color Texture::at(Point3 position) {
    Color black{};

    // TODO: wrapping?
    if(position.x() < 0.0 || position.x() > 1.0)
        return black;
    if(position.y() < 0.0 || position.y() > 1.0)
        return black;

    int px = static_cast<int>(position.x() * (width-1));
    int py = static_cast<int>(position.y() * (height-1));
    return pixels[py*width+px];
}

__global__
void compileTexture(Texture** destination, int w, int h, Color* pixels) {
    *destination = new Texture(w, h, pixels);
}

__host__ Texture** Texture::loadFromFile(const std::string& name) {
    std::cout << "Loading texture " << name << std::endl;
    int width;
    int height;
    int channels;
    uint8_t* pixels = stbi_load(name.c_str(), &width, &height, &channels, 3);
    Color* colors;
    checkCudaErrors(cudaMallocManaged(&colors, sizeof(Color)*width*height));

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            size_t ptr = (x + width*y)*3;
            int red = pixels[ptr+0];
            int green = pixels[ptr+1];
            int blue = pixels[ptr+2];
            colors[ptr/3] = Color(double(red)/255.0, double(green)/255.0, double(blue)/255.0);
        }
    }

    Texture** result;
    checkCudaErrors(cudaMallocManaged(&result, sizeof(Texture*)));

    compileTexture<<<1, 1>>>(result, width, height, colors);
    checkCudaErrors(cudaDeviceSynchronize());

    return result;
}



