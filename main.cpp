#include "rtweekend.h"

#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "moving_sphere.h"
#include "aarect.h"
#include "box.h"
#include "bvh.h"
#include "pdf.h"

#include <iostream>
#include <string>
#include <fstream>
#include <time.h>

auto aspect_ratio = 1.0 / 1.0;
int image_width = 512;
int image_height = static_cast<int>(image_width / aspect_ratio);
int samples_per_pixel = 200;
int max_depth = MAX_DEPTH;
bool fixed = false;
//bool fixed = true;
bool rotate = false;

point3 lookfrom(278, 278, -800);
point3 lookat(278, 278, 0);
vec3 vup(0, 1, 0);
auto dist_to_focus = 10.0;
auto aperture = 0.0;
auto vfov = 40.0;
auto t0 = 0.0;
auto t1 = 1.0;

int currFrame = 0;
int MAX_FRAMES = 0;
std::string outputFileName = "";

cl_double3 double3(const vec3& v2) {
    return { v2.x(), v2.y(), v2.z() };
}
void Wait(cl_command_queue queue) {
    cl_event wait;
    cl_int status;

    status = clEnqueueMarker(queue, &wait);
    status = clWaitForEvents(1, &wait);
}
//To render this scene, change the background in kernel.cl and most importantly, change the 
//world and mats array size from 20 to 500
hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.55) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.75) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.82) {
                    // glass
                    sphere_material = make_shared<dielectric>(color(1, 1, 1), 1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else {
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<diffuse_light>(15*albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(color(1, 1, 1), 1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    lookfrom = vec3(13, 2, 3);
    lookat = (1-0.74)*lookfrom;
    //lookfrom = 0.9 * lookfrom;
    vup = vec3(0, 1, 0);
    dist_to_focus = 10.0;
    aperture = 0.1;
    vfov = 25.0;
    t0 = 0.0;
    t1 = 1.0;
    outputFileName = "animation/NoRL/random_scene";
    return world;
}
hittable_list teapot_scene() {
    hittable_list world;

    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto blue = make_shared<lambertian>(color(0.0 / 255, 188.0 / 255, 212.0 / 255));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));
    auto glass = make_shared<dielectric>(color(1, 1, 1), 1.5);
    auto true_white = make_shared<lambertian>(color(.9, .9, .9));
    auto orange = make_shared<lambertian>(color(1.0, 64.0 / 255, 0.0));
    auto brown = make_shared<lambertian>(color(222.0 / 255, 184.0 / 255, 135.0 / 255));
    auto gold = make_shared<lambertian>(color(230.0 / 255, 172.0 / 255, 0.0));
    color metal_color(230.0 / 255, 172.0 / 255, 0.0);
    shared_ptr<material> aluminum = make_shared<metal>(metal_color, 0.0);
    double R = 120;
    double zBack = -200;
    world.add(make_shared<yz_rect>(0, 700, zBack, 555, 455 + 277, green));
    world.add(make_shared<yz_rect>(0, 700, zBack, 555, 0 - 377, red));
    world.add(make_shared<sphere>(vec3(-377 + 3 * R + 50, R, 380), R, aluminum));
    world.add(make_shared<xz_rect>(0 - 377, 455 + 277, zBack, 555, 700, white));
    world.add(make_shared<xz_rect>(0 - 377, 455 + 277, zBack, 555, 0, white));
    world.add(make_shared<xy_rect>(0 - 377, 455 + 277, 0, 700, 555, white));

    world.add(make_shared<xy_rect>(0, 400, 0, 555, 555, brown)); //Door
    rotate = true;
    world.add(make_shared<xy_rect>(400, 610, 0, 555, 555, light));
    world.add(make_shared<xy_rect>(610, 455 + 277, 0, 555, 555, white));
    //world.add(make_shared<xy_rect>(0 - 377, 455 + 277, 0, 700, zBack, white)); //CameraBack
    world.add(make_shared<sphere>(vec3(-377 + 2 * R + 50, 2.732 * R, 380), R, red));
    world.add(make_shared<sphere>(vec3(-377 + R + 50, R, 380), R, orange));
    //world.add(make_shared<sphere>(vec3(-377 + R + (1109-2*R)*(3*1.0/4), 350, 135), R, glass));

    lookfrom = vec3(177, 350, -750);
    lookat = vec3(177, 350, 0);
    vup = vec3(0, 1, 0);
    dist_to_focus = 10.0;
    aperture = 0.0;
    vfov = 55.0;
    t0 = 0.0;
    t1 = 1.0;
    aspect_ratio = 4.0 / 3.0;
    image_height = static_cast<int>(image_width / aspect_ratio);
    outputFileName = "animation/NoRL/teapotBidirTest";

    return world;
}
hittable_list cornell_box() {
    hittable_list world;

    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto blue = make_shared<lambertian>(color(0.0 / 255, 188.0 / 255, 212.0 / 255));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));
    auto glass = make_shared<dielectric>(color(1, 1, 1), 1.5);
    auto glass1 = make_shared<dielectric>(color(223.0 / 255, 128.0 / 255, 1.0), 1.5);
    auto true_white = make_shared<lambertian>(color(.9, .9, .9));
    auto orange = make_shared<lambertian>(color(1.0, 64.0 / 255, 0.0));
    double scale = 1.0;
    double zBack = -528;
    //point3 offset(130 * scale,168 * scale,-30 * scale);
    point3 offset(0, 0, 0);
    world.add(make_shared<yz_rect>(-278 * scale + offset.y(), 278 * scale + offset.y(), zBack + offset.z(), 328 * scale + offset.z(), 278 * scale + offset.x(), green));
    world.add(make_shared<yz_rect>(-278 * scale + offset.y(), 278 * scale + offset.y(), zBack + offset.z(), 328 * scale + offset.z(), -278 * scale + offset.x(), red));
    world.add(make_shared<xz_rect>(-75 * scale + offset.x(), 75 * scale + offset.x(), -65 * scale + offset.z(), 65 * scale + offset.z(), 277 * scale + offset.y(), light));
    world.add(make_shared<xz_rect>(-278 * scale + offset.x(), 278 * scale + offset.x(), zBack + offset.z(), 328 * scale + offset.z(), 278 * scale + offset.y(), white)); //ceiling
    world.add(make_shared<xz_rect>(-278 * scale + offset.x(), 278 * scale + offset.x(), zBack + offset.z(), 328 * scale + offset.z(), -278 * scale + offset.y(), white)); //floor
    world.add(make_shared<xy_rect>(-278 * scale + offset.x(), 278 * scale + offset.x(), -278 * scale + offset.y(), 278 * scale + offset.y(), 328 * scale + offset.z(), white));
    //world.add(make_shared<xy_rect>(-75 + offset.x(), 75 + offset.x(), 128 + offset.y(), 278 + offset.y(), -70 + offset.z(), white)); //LightBlock
    //world.add(make_shared<xy_rect>(-278 + offset.x(), 278 + offset.x(), -278 + offset.y(), 278 + offset.y(), zBack + offset.z(), white)); //BackWall

    //color metal_color(230.0/255, 172.0/255, 0.0);
    color metal_color(0.8, 0.85, 0.88);
    shared_ptr<material> aluminum = make_shared<metal>(metal_color, 0.0);
    //world.add(make_shared<sphere>(vec3(101 , 100 , 405 ), 100 , red));
    world.add(make_shared<sphere>(vec3(-130 * scale + offset.x(), -168 * scale + offset.y(), 30 * scale + offset.z()), 110 * scale, glass));
    world.add(make_shared<sphere>(vec3(147 * scale + offset.x(), -148 * scale + offset.y(), 198 * scale + offset.z()), 130 * scale, aluminum));
    
    lookfrom = vec3(0 * scale, 0 * scale, -1070 * scale) + offset;
    lookat = vec3(0 * scale, 0 * scale, 0 * scale) + offset;
    vup = vec3(0, 1, 0);
    dist_to_focus = 10.0;
    aperture = 0.0;
    vfov = 40.0;
    t0 = 0.0;
    t1 = 1.0;
    outputFileName = "output/frame";
    return world;
}
hittable_list low_lighting() {
    hittable_list world;

    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto blue = make_shared<lambertian>(color(0.25, 0.25, 0.75));
    auto light = make_shared<diffuse_light>(color(10, 10, 10));
    auto glass = make_shared<dielectric>(color(1, 1, 1), 1.5);
    double scale = 1.0;
    double zBack = -428;
    double lightWidth = 50 * scale;
    //point3 offset(130 * scale,168 * scale,-30 * scale);
    point3 offset(0, 0, 0);
    world.add(make_shared<yz_rect>(-278 * scale + offset.y(), 278 * scale + offset.y(), zBack + offset.z(), 328 * scale + offset.z(), 278 * scale + offset.x(), green));
    world.add(make_shared<yz_rect>(-278 * scale + offset.y(), 278 * scale + offset.y(), zBack + offset.z(), 328 * scale + offset.z(), -278 * scale + offset.x(), red));
    //world.add(make_shared<xz_rect>(-75 * scale + offset.x(), 75 * scale + offset.x(), -65 * scale + offset.z(), 65 * scale + offset.z(), 277 * scale + offset.y(), light));
    world.add(make_shared<xz_rect>(-278 * scale + offset.x(), 278 * scale + offset.x(), zBack + offset.z(), 328 * scale + offset.z(), 278 * scale + offset.y(), white)); //ceiling
    world.add(make_shared<xz_rect>(-278 * scale + offset.x(), 278 * scale + offset.x(), zBack + offset.z(), 328 * scale + offset.z(), -278 * scale + offset.y(), white)); //floor
    world.add(make_shared<xy_rect>(-278 * scale + offset.x(), 278 * scale + offset.x(), -278 * scale + offset.y(), 278 * scale + offset.y(), 328 * scale + offset.z(), white));
    //world.add(make_shared<sphere>(vec3(-230 * scale, 0, 0) + offset, 25 * scale, light));
    world.add(make_shared<yz_rect>(-lightWidth + offset.y(), lightWidth + offset.y(), -lightWidth + offset.z(), lightWidth + offset.z(), -277 * scale + offset.x(), light));
    double wallOff = 0 * scale;
    double wallwidth = 10 * scale;
    double wallheight = 250 * scale;
    double wallback = 300 * scale;
    double wallFront = -300 * scale;
    world.add(make_shared<yz_rect>(-wallheight + offset.y(), wallheight + offset.y(), wallFront + offset.z(), wallback + offset.z(), wallOff - wallwidth + offset.x(), blue));
    world.add(make_shared<yz_rect>(-wallheight + offset.y(), wallheight + offset.y(), wallFront + offset.z(), wallback + offset.z(), wallOff + wallwidth + offset.x(), blue));
    world.add(make_shared<xy_rect>(-wallwidth + wallOff + offset.x(), wallwidth + wallOff + offset.x(), -wallheight + offset.y(), wallheight + offset.y(), wallFront + offset.z(), blue));
    world.add(make_shared<xy_rect>(-wallwidth + wallOff + offset.x(), wallwidth + wallOff + offset.x(), -wallheight + offset.y(), wallheight + offset.y(), wallback + offset.z(), blue));
    world.add(make_shared<xz_rect>(-wallwidth + wallOff + offset.x(), wallwidth + wallOff + offset.x(), wallFront + offset.z(), wallback * scale + offset.z(), wallheight + offset.y(), white));
    world.add(make_shared<xz_rect>(-wallwidth + wallOff + offset.x(), wallwidth + wallOff + offset.x(), wallFront + offset.z(), wallback * scale + offset.z(), -wallheight + offset.y(), white));
    //world.add(make_shared<xy_rect>(-278 + offset.x(), 278 + offset.x(), -278 + offset.y(), 278 + offset.y(), zBack + offset.z(), white)); //BackWall

    color metal_color(0.8, 0.85, 0.88);
    shared_ptr<material> aluminum = make_shared<metal>(metal_color, 0.0);
    //world.add(make_shared<sphere>(vec3(101 , 100 , 405 ), 100 , red));
    world.add(make_shared<sphere>(vec3(120 * scale + offset.x(), 110 * scale + offset.y(), 0 * scale + offset.z()), 60 * scale, blue));
    world.add(make_shared<sphere>(vec3(120 * scale + offset.x(), -110 * scale + offset.y(), 0 * scale + offset.z()), 60 * scale, aluminum));
    world.add(make_shared<sphere>(vec3(-100 * scale + offset.x(), 110 * scale + offset.y(), 0 * scale + offset.z()), 60 * scale, blue));
    world.add(make_shared<sphere>(vec3(-100 * scale + offset.x(), -110 * scale + offset.y(), 0 * scale + offset.z()), 60 * scale, aluminum));

    lookfrom = vec3(0 * scale, 0 * scale, -1600 * scale) + offset;
    lookat = vec3(0 * scale, 0 * scale, 0 * scale) + offset;
    vup = vec3(0, 1, 0);
    dist_to_focus = 10.0;
    aperture = 0.0;
    vfov = 25.0;
    t0 = 0.0;
    t1 = 1.0;
    outputFileName = "animation/noiseFreelowlighting";
    return world;
}
void checkAndReport(bool cond, std::string message) {
    if (cond) {
        std::cout << message << "\n";
        exit(0);
    }
}
int clRender(const std::string& fileName) {
    clock_t tStart = clock();
    //The scene to be rendered can be changed here
    //auto world = teapot_scene();
    auto world = cornell_box();
    //auto world = random_scene();
    //auto world = low_lighting();
    int	NUM_ELEMENTS = image_width * image_height;

    std::string strFileName = fileName;
    const char* CL_FILE_NAME = { "kernel.cl" };
    void Wait(cl_command_queue);

    FILE* fp;
    fp = fopen(CL_FILE_NAME, "rb");

    cl_platform_id platform;
    cl_device_id device;

    cl_int status = clGetPlatformIDs(1, &platform, NULL);
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    int depth_val = MAX_DEPTH, numObjects = world.objects.size();
    cl_double4* hA = new cl_double4[NUM_ELEMENTS]; // Output Color
    cl_int2* hB = new cl_int2[NUM_ELEMENTS]; // Dimensions
    cl_double16* hC = new cl_double16[NUM_ELEMENTS]; // Camera
    cl_double8* hE = new cl_double8[numObjects]; //  Object List
    cl_int* hF = new cl_int[1]; // Object List Size
    cl_double4* hH = new cl_double4[NUM_ELEMENTS]; // Materials List
    cl_int* hK = new cl_int[2]; //Samples Per Pixel
    cl_int* hL = new cl_int[numObjects]; //Offset into points array

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    cl_command_queue cmdQueue = clCreateCommandQueue(context, device, 0, &status);

    //cl_mem dA = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_t(NUM_ELEMENTS * sizeof(cl_double4)), NULL, &status);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(NUM_ELEMENTS * sizeof(cl_int2)), NULL, &status);
    cl_mem dC = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(NUM_ELEMENTS * sizeof(cl_double16)), NULL, &status);
    cl_mem dE = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(numObjects * sizeof(cl_double8)), NULL, &status);
    cl_mem dF = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(1 * sizeof(cl_int)), NULL, &status);
    cl_mem dH = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(NUM_ELEMENTS * sizeof(cl_double4)), NULL, &status);
    cl_mem dK = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(2 * sizeof(cl_int)), NULL, &status);
    cl_mem dL = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(numObjects * sizeof(cl_int)), NULL, &status);
    int qtableSize;


    int hJctr = 0;
    hB[0] = { image_width,image_height };
    hC[0] = { lookfrom.x(), lookfrom.y(), lookfrom.z(),
                lookat.x(), lookat.y(), lookat.z(),
                vup.x(), vup.y(), vup.z(),
                vfov, aperture, dist_to_focus, t0, t1, 0.0, 0.0 };
    hF[0] = numObjects;
    hK[0] = samples_per_pixel;

    int totalPoints = 0; //Total number of Points
    for (int i = 0; i < numObjects; i++) {
        std::vector<double> info = world.objects[i]->getInfo();
        int cType = world.objects[i]->clType();

        int curr = 0;
        if (fixed) { //fixed or variable number of points on the surface
            curr = NUM_POINTS;
        }
        else if (cType == 1 || cType == 2 || cType == 3) {
            if (USE_HAMM == 1) {
                double ratio = sqrt((info[1] - info[0]) * (info[3] - info[2]) / (555.0 * 555.0));
                curr = std::max(1, (int)(NUM_POINTS * ratio));
            }
            else {
                double ratio = sqrt(sqrt((info[1] - info[0]) * (info[3] - info[2]) / (555.0 * 555.0)));
                curr = SQRTN * ratio * SQRTN * ratio;
            }
        }
        else if (cType == 5) {
            if (USE_HAMM == 1) {
                double ratio = sqrt(4 * pi * info[3] * info[3] / (555.0 * 555.0));
                curr = std::max(1, (int)(NUM_POINTS * ratio));
            }
            else {
                double ratio = sqrt(sqrt(4 * pi * info[3] * info[3] / (555.0 * 555.0)));
                curr = SQRTN * ratio * SQRTN * ratio;
            }
        }
        else {
            curr = NUM_POINTS;
        }
        totalPoints += curr;
    }

    cl_int err = CL_SUCCESS;
    cl_device_svm_capabilities caps;

    //Setup Coarse grained Shared Virtual Memory for Qtable
    double val = NUM_ACTIONS;
    err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &caps, 0);
    checkAndReport(!(err == CL_SUCCESS && (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)), "Device not Compatible");
    qtableSize = totalPoints * NUM_ACTIONS;
    cl_double* dI = (cl_double*)clSVMAlloc(context, CL_MEM_READ_WRITE, qtableSize * sizeof(cl_double), 0);
    checkAndReport(!dI, "Wrong1");
    err = clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_WRITE, dI, qtableSize * sizeof(cl_double), 0, 0, 0);
    checkAndReport(err != CL_SUCCESS, "Wrong2");
    //memset(dI, 1.0/val, qtableSize * sizeof(cl_double));
    for (int i = 0; i < qtableSize; i++)
        dI[i] = 1.0 / val;
    err = clEnqueueSVMUnmap(cmdQueue, dI, 0, 0, 0);
    checkAndReport(err != CL_SUCCESS, "Wrong3");

    //Declare dA
    cl_double4* dA = (cl_double4*)clSVMAlloc(context, CL_MEM_READ_WRITE, size_t(NUM_ELEMENTS * sizeof(cl_double4)), 0);
    checkAndReport(!dA, "Wrong1");

    cl_double2* hJ = new cl_double2[totalPoints];
    cl_mem dJ = clCreateBuffer(context, CL_MEM_READ_ONLY, size_t(totalPoints * sizeof(cl_double2)), NULL, &status);

    fseek(fp, 0, SEEK_END);
    size_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* clProgramText = new char[fileSize + 1];
    size_t n = fread(clProgramText, 1, fileSize, fp);
    clProgramText[fileSize] = '\0';
    fclose(fp);

    char* strings[1];
    strings[0] = clProgramText;
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)strings, NULL, &status);
    delete[] clProgramText;

    char* options = {};
    status = clBuildProgram(program, 1, &device, options, NULL, NULL);

    char* str;
    size_t sstr;
    status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &sstr);
    str = (char*)malloc(sstr);
    status |= clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sstr, str, NULL);
    printf("\n%s \n", str);
    free(str);
    for (currFrame = 0; currFrame <= MAX_FRAMES; currFrame++) {
        clock_t frameStart = clock();
        std::cout << "\nFrame: " << currFrame << "\n";
        //world = teapot_scene();
        world = cornell_box();
        //world = low_lighting();
        int points_so_far = 0; //Total number of Points
        
        //samples_per_pixel = currFrame == 0 ? 10 : 200;
        /*if (currFrame == 0)
            samples_per_pixel = 10;
        else
            samples_per_pixel = 50 * currFrame;*/
        hK[0] = samples_per_pixel;
        hK[1] = currFrame == 0 ? 1 : 0; //do redundant sampling
        for (int i = 0; i < numObjects; i++) {
            std::vector<double> info = world.objects[i]->getInfo();
            int cType = world.objects[i]->clType();

            if (cType == 1 || cType == 2 || cType == 3 || cType == 4 || cType == 5) {
                if (i == 6 && rotate)
                    hE[i] = { info[0], info[1], info[2], info[3], info[4], info[5], 20.0,cl_double(cType) };
                else
                    hE[i] = { info[0], info[1], info[2], info[3], info[4], info[5], 0.0,cl_double(cType) };
            }

            hH[i] = { info[6],info[7],info[8],info[9] };

            int curr = 0;
            if (fixed) { //fixed or variable number of points on the surface
                hL[i] = SQRTN;
                curr = NUM_POINTS;
            }
            else if (cType == 1 || cType == 2 || cType == 3) {
                if (USE_HAMM == 1) {
                    double ratio = sqrt((info[1] - info[0]) * (info[3] - info[2]) / (555.0 * 555.0));
                    curr = std::max(1, (int)(NUM_POINTS * ratio));
                }
                else {
                    double ratio = sqrt(sqrt((info[1] - info[0]) * (info[3] - info[2]) / (555.0 * 555.0)));
                    hL[i] = SQRTN * ratio;
                    curr = hL[i] * hL[i];
                }
            }
            else if (cType == 5) {
                if (USE_HAMM == 1) {
                    double ratio = sqrt(4 * pi * info[3] * info[3] / (555.0 * 555.0));
                    curr = std::max(1, (int)(NUM_POINTS * ratio));
                }
                else {
                    double ratio = sqrt(sqrt(4 * pi * info[3] * info[3] / (555.0 * 555.0)));
                    hL[i] = SQRTN * ratio;
                    curr = hL[i] * hL[i];
                }
            }
            else {
                hL[i] = SQRTN;
                curr = NUM_POINTS;
            }
            points_so_far += curr;
            if (USE_HAMM == 1)
                hL[i] = points_so_far;
        }
        if (USE_HAMM == 1) {
            for (int i = 0; i < numObjects; i++) {
                if (i == 0)
                    getHammerselyPoints1(hJ, hL[i]);
                else
                    getHammerselyPoints1(&hJ[hL[i - 1]], hL[i] - hL[i - 1]);
            }
        }


        //status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, size_t(NUM_ELEMENTS * sizeof(cl_double4)), hA, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dB, CL_FALSE, 0, size_t(NUM_ELEMENTS * sizeof(cl_int2)), hB, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dC, CL_FALSE, 0, size_t(NUM_ELEMENTS * sizeof(cl_double16)), hC, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dE, CL_FALSE, 0, size_t(numObjects * sizeof(cl_double8)), hE, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dF, CL_FALSE, 0, size_t(1 * sizeof(cl_int)), hF, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dH, CL_FALSE, 0, size_t(NUM_ELEMENTS * sizeof(cl_double4)), hH, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dJ, CL_FALSE, 0, size_t(points_so_far * sizeof(cl_double2)), hJ, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dK, CL_FALSE, 0, size_t(2 * sizeof(cl_int)), hK, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(cmdQueue, dL, CL_FALSE, 0, size_t(numObjects * sizeof(cl_int)), hL, 0, NULL, NULL);

        Wait(cmdQueue);

        cl_kernel kernel = clCreateKernel(program, "Render", &status);

        /*size_t maxWorkGroupSize;
        err = clGetKernelWorkGroupInfo(kernel,device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
        checkAndReport(err != CL_SUCCESS, "Wrong0");
        std::cout << "\nMax Work Group Size: " << maxWorkGroupSize << std::endl;*/

        //status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        err = clSetKernelArgSVMPointer(kernel, 0, dA);
        checkAndReport(err != CL_SUCCESS, "Wrong4");
        err = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(dA), &dA);
        checkAndReport(err != CL_SUCCESS, "Wrong5");

        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
        status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dE);
        status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &dF);
        status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &dH);
        status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &dJ);
        status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &dK);
        status = clSetKernelArg(kernel, 8, sizeof(cl_mem), &dL);
        err = clSetKernelArgSVMPointer(kernel, 9, dI);
        checkAndReport(err != CL_SUCCESS, "Wrong4");
        err = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(dI), &dI);
        checkAndReport(err != CL_SUCCESS, "Wrong5");

        size_t globalWorkSize[1] = { size_t(NUM_ELEMENTS) };
        size_t localWorkSize[1] = { size_t(256) };

        Wait(cmdQueue);
        std::cout << "Computing...\n";
        std::string frameNum = std::to_string(currFrame);
        std::ofstream ofImage(strFileName + frameNum + (std::string)".ppm"); // Open Image File
        if (!ofImage.is_open()) {
            std::cout << "Could not Open Image\n";
            return -1;
        }
        ofImage << "P3\n" << image_width << " " << image_height << "\n255\n"; // PPM Header with dimensions and color index
        err = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        checkAndReport(err != CL_SUCCESS, "Crashed");
        Wait(cmdQueue);

        //status = clEnqueueReadBuffer(cmdQueue, dA, CL_TRUE, 0, size_t(NUM_ELEMENTS * sizeof(cl_double4)), hA, 0, NULL, NULL);
        /*clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_WRITE, dI, qtableSize * sizeof(cl_double), 0, 0, 0);
        for (size_t j = 0; j < 10; j++)
            std::cout << dI[j] << "\n";
        clEnqueueSVMUnmap(cmdQueue, dI, 0, 0, 0);*/
        clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_WRITE, dA, size_t(NUM_ELEMENTS * sizeof(cl_double4)), 0, 0, 0);

        for (int i = NUM_ELEMENTS - 1; i >= 0; i--) {

            double ix = 1 + (i / image_width);
            int curY = int(floor(ix));
            int curX = image_width - (i - ((curY - 1) * image_width)) - 1;

            int curInt = ((curY - 1) * image_width) + curX;
            write_color(ofImage, color(dA[curInt].x, dA[curInt].y, dA[curInt].z), samples_per_pixel);
        }
        clEnqueueSVMUnmap(cmdQueue, dA, 0, 0, 0);

        ofImage.close(); // Close image file

        clReleaseKernel(kernel);
           
        clock_t frameEnd = clock();
        std::cout << "\nRender Time: " << (double)(frameEnd - frameStart) / CLOCKS_PER_SEC << " seconds\n";
    }
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    //clReleaseMemObject(dA);
    clSVMFree(context, dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseMemObject(dE);
    clReleaseMemObject(dF);
    clReleaseMemObject(dH);
    clReleaseMemObject(dJ);
    clReleaseMemObject(dK);
    clReleaseMemObject(dL);
    clSVMFree(context, dI);

    delete[] hA, hB, hC, hE, hF, hH, hJ, hK, hL;
    std::cout << "\nTime Elapsed: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << " seconds\n";
    return 0;
}

bool show = true;
color ray_color(
    ray& r,
    const color& background,
    const hittable& world,
    shared_ptr<hittable> light_shape
) {
    int idxQ = 0, idxPrev;
    color lastAttenuation, ans(1, 1, 1);
    double** prevQtable = NULL;
    double* prevQmax = NULL;
    for (int depth = 0; depth <= MAX_DEPTH; depth++) {
        hit_record rec;

        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth >= MAX_DEPTH) {
            ans = ans * color(0, 0, 0);
            break;
        }
        // If the ray hits nothing, return the background color.
        if (!world.hit(r, 0.001, infinity, rec)) {
            ans = ans * background;
            break;
        }

        ray scattered;
        color attenuation;
        color emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
        color update;
        int idxCurr = 0;
        double sum = 0;
        if (RL_ON == 1) {
            idxCurr = getClosestIndex(rec.hammersely_points, NUM_POINTS, rec.p);
            if (depth > 0) {
                if (rec.mat_ptr->MatType() == 1) {
                    update = emitted;
                }
                else {
                    int qmaxid = 0;
                    for (int i = 0; i < NUM_ACTIONS; i++) {
                        if (rec.qtable[idxCurr][i] > rec.qtable[idxCurr][qmaxid])
                            qmaxid = i;
                    }
                    double qupdate = rec.qtable[idxCurr][qmaxid];
                    update = lastAttenuation * qupdate;
                    // update = lastAttenuation*rec.qmax[idxCurr];
                }
                double lr = double_min(0.3, double_max(1.0 / (1 + rec.visits[idxPrev][idxQ]), 0.05));
                // double lr = 0.2;
                double update_val = update.max_value();
                // update_val = dot(update,vec3(0.2989,0.5870,0.114)); //rgb to gray
                prevQtable[idxPrev][idxQ] = (1 - lr) * prevQtable[idxPrev][idxQ] + lr * update_val;
                rec.visits[idxPrev][idxQ]++;
                // if (prevQtable[idxPrev][idxQ]>prevQmax[idxPrev])
                //     prevQmax[idxPrev] = prevQtable[idxPrev][idxQ];
            }
            double scattercdf[NUM_ACTIONS];

            for (int i = 0; i < NUM_ACTIONS; i++) {
                sum += rec.qtable[idxCurr][i];
                scattercdf[i] = sum;
            }
            double temprand = random_double() * sum;
            for (idxQ = 0; idxQ < NUM_ACTIONS; idxQ++) {
                if (temprand <= scattercdf[idxQ])
                    break;
            }
        }
        double pdf_val;
        color albedo;
        if (!rec.mat_ptr->scatter(r, rec, albedo, scattered, pdf_val, idxQ)) {
            ans = ans * emitted;
            break;
        }
        if (RL_ON == 1) {
            double numPatches = NUM_ACTIONS;
            pdf_val = (rec.qtable[idxCurr][idxQ] / sum) * numPatches / (2 * pi);
        }
        if (RL_ON == 0) {
            auto p0 = make_shared<hittable_pdf>(light_shape, rec.p);
            auto p1 = make_shared<cosine_pdf>(rec.normal, idxQ);
            mixture_pdf p(p0, p1);

            scattered = ray(rec.p, p.generate(), r.time());
            pdf_val = p.value(scattered.direction());
            //pdf_val = 0.5 * pdf_val + p.value(scattered.direction());
        }

        if (rec.mat_ptr->MatType() == 2 || rec.mat_ptr->MatType() == 3) {
            attenuation = albedo;
            if (LS==0)
                pdf_val = 1;
        }
        else {
            double spdf = rec.mat_ptr->scattering_pdf(r, rec, scattered);
            attenuation = albedo * spdf;
        }
        ans = ans * attenuation / pdf_val;
        lastAttenuation = attenuation;
        prevQtable = rec.qtable;
        prevQmax = rec.qmax;
        idxPrev = idxCurr;
        r = scattered;
    }
    return ans;
}
int main() {
    /*for (int i = 10; i <= 600; i+=50) {
        samples_per_pixel = i;
        std::cout << "*************\nSamples per pixel = "<<samples_per_pixel<<"\n";
        std::string tmp(1,(char)(i/50+48));
        if (i == 600)
            tmp = "12";
        else if (i == 550)
            tmp = "11";
        else if (i == 500)  
            tmp = "10";
        if (i == 10) i = 400;
        clRender("render"+tmp);
    }*/
    std::cout << "Samples per pixel = " <<samples_per_pixel<<"\n";
    /*if (RL_ON == 1)
        clRender("animation/RL/frame_215");
    else*/
        clRender(outputFileName);
    std::cout << "Finished...\n";
    return 0;

    // World
    clock_t tStart = clock();
    shared_ptr<hittable> light_shape = make_shared<xz_rect>(213, 343, 227, 332, 554, shared_ptr<material>());

    auto world = cornell_box();

    color background(0, 0, 0);

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, t0, t1);
    // Render
    std::ofstream outfile("render.ppm");
    outfile << "P3\n" << image_width << " " << image_height << "\n255\n";
    std::vector<std::vector<color> > out_arr(image_height,std::vector<color>(image_width));
    #pragma omp parallel for
    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                double** temp;
                pixel_color += ray_color(r, background, world, light_shape);
            }
            out_arr[j][i] = pixel_color;
        }
    }
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i)
            write_color(outfile, out_arr[j][i], samples_per_pixel);
    }
    std::cout << "\nTime Elapsed: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << " seconds\n";
    std::cerr << "\nDone.\n";
}