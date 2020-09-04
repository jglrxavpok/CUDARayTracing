#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <math.h>
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
static constexpr int IMAGE_WIDTH = 200;
static constexpr int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH/ASPECT_RATIO);
static constexpr int MAX_BOUNCE = 50;

auto materialGround = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
auto materialCenter = std::make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
auto materialLeft   = std::make_shared<Dielectric>(1.5);
auto materialRight  = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 0.0);

IntersectableGroup group{
    std::make_shared<Sphere>(Point3( 0.0, -100.5, -1.0), 100.0, materialGround),
    std::make_shared<Sphere>(Point3( 0.0,    0.0, -1.0), 0.5, materialCenter),
    std::make_shared<Sphere>(Point3(-1.0,    0.0, -1.0), 0.5, materialLeft),
    std::make_shared<Sphere>(Point3(-1.0,    0.0, -1.0), -0.45, materialLeft),
    std::make_shared<Sphere>(Point3( 1.0,    0.0, -1.0), 0.5, materialRight),
};

Color trace(const Ray& r, const Intersectable& world, int remainingRays = MAX_BOUNCE) {
    static Color skyBlue = Color(0.5, 0.7, 1.0);
    static Color white = Color(1.0, 1.0, 1.0);
    static Color red = Color(1.0, 0.0, 0.0);
    static Color black = Color(0.0, 0.0, 0.0);

    if(remainingRays <= 0) {
        return black;
    }

    HitResult result{};
    if(world.hit(r, 0.001 /* remove shadow acne */, INFINITY, result)) {
        Ray scattered{};
        Color attenuation{};
        if(result.material->scatter(r, result, attenuation, scattered)) {
            return attenuation * trace(scattered, world, remainingRays-1);
        }
        return white;
    }

    Vec3 direction = r.direction().normalized();
    // map from -1..1 to 0..1
    auto alpha = (direction.y() + 1.0) / 2.0;
    return (1.0-alpha) * white + alpha * skyBlue;
}

int main()
{
    Camera camera{Point3(-2,2,1), Point3(0,0,-1), Vec3(0,1,0), 20, ASPECT_RATIO};
    uint8_t* pixels = new uint8_t[IMAGE_WIDTH*IMAGE_HEIGHT*4];

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        std::cout << "Remaining scanlines: " << IMAGE_HEIGHT-y << std::endl;
        for(int x = 0; x < IMAGE_WIDTH; x++) {
            Color pixelColor{};
            for(int i = 0; i < Camera::SAMPLES_PER_PIXEL; i++) {
                double u = double(x+randomDouble()) / (IMAGE_WIDTH-1);
                double v = double(y+randomDouble()) / (IMAGE_HEIGHT-1);

                pixelColor += trace(camera.generateRay(u, v), group);
            }

            int yInversed = IMAGE_HEIGHT-y-1;
            size_t ptr = (yInversed*IMAGE_WIDTH+x)*4;
            writeColor(pixels, ptr, pixelColor, Camera::SAMPLES_PER_PIXEL);
        }

        stbi_write_png("./test.png", IMAGE_WIDTH, IMAGE_HEIGHT, 4, pixels, IMAGE_WIDTH*4);
    }

    stbi_write_png("./test.png", IMAGE_WIDTH, IMAGE_HEIGHT, 4, pixels, IMAGE_WIDTH*4);
    delete[] pixels;
    return 0;
}