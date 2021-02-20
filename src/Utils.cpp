/*
 * Utils.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "Utils.h"
#include <iostream>

#include "cuda_runtime_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS Image_channels

using namespace std;

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

Image* loadJPEG(const char *filename) {
	Image* img = NULL;
	int width, height, bpp;

	unsigned char* rgb_image = stbi_load(filename, &width, &height, &bpp, CHANNELS);

	if (rgb_image != NULL) {
		img = Image_new(width, height, CHANNELS, rgb_image);
	}
	else {
		// TODO add error propagation
	}

	return img;
}

SetOfPoints pixelize(const Image* img) {
	int height = Image_getHeight(img);
	int width = Image_getWidth(img);
    int channels = Image_getChannels(img);

    SetOfPoints data;
    //data.pointList = (Point *) calloc(width * height, sizeof (Point));
    CUDA_CHECK_RETURN(cudaMallocHost((void**)&(data.pointList), width * height * sizeof(Point), cudaHostAllocMapped));
    data.sizeList = width * height;

    float* coordinates;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            //coordinates = (float *) calloc(channels, sizeof(float));
        	CUDA_CHECK_RETURN(cudaMallocHost((void**)&coordinates, channels * sizeof(float), cudaHostAllocMapped));
            for (int z = 0; z < channels; z++) {
            	coordinates[z] = (float) (Image_getData(img)[y * width * channels + x * channels + z]);
            }
            insertPoint(&data,
                        Point{.coordinates = coordinates, .dimension = channels, .metadata = to_string(x) + "," + to_string(y)},
                        y * width + x);
        }
    }

    return data;
}

bool savePNG(SetOfPoints* clusters, int k, const char *filename, int width, int height) {
	unsigned char* rgb_image = (unsigned char*) malloc(sizeof(unsigned char) * width * height * CHANNELS);

    for (int i = 0; i < k; i++) {
        SetOfPoints cluster = clusters[i];

        Point center = getCenter(cluster);
        unsigned char r = (unsigned char) clamp(getCoordinateAt(center, 0), 0, 255);
        unsigned char g = (unsigned char) clamp(getCoordinateAt(center, 1), 0, 255);
        unsigned char b = (unsigned char) clamp(getCoordinateAt(center, 2), 0, 255);
        Point_delete(&center);

        for (int j = 0; j < cluster.sizeList; j++) {
            string position = cluster.pointList[j].metadata;
            int splitIndex = position.find_first_of(',', 0); //WORKS ONLY FOR 2D IMAGES
            int x = stoi(position.substr(0, splitIndex));
            int y = stoi(position.substr(splitIndex + 1, position.length() - 1));

            rgb_image[y * width * CHANNELS + x * CHANNELS] = r;
            rgb_image[y * width * CHANNELS + x * CHANNELS + 1] = g;
            rgb_image[y * width * CHANNELS + x * CHANNELS + 2] = b;
        }
    }

	return stbi_write_png(filename, width, height, CHANNELS, (void*)rgb_image, width * CHANNELS) ? true : false;
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     	 case 2: // Fermi
     		 if (devProp.minor == 1) cores = mp * 48;
     		 else cores = mp * 32;
     		 break;
     	 case 3: // Kepler
     		 cores = mp * 192;
     		 break;
     	 case 5: // Maxwell
     		 cores = mp * 128;
     		 break;
     	 case 6: // Pascal
     		 if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
     		 else if (devProp.minor == 0) cores = mp * 64;
     		 else printf("Unknown device type\n");
     		 break;
     	 case 7: // Volta and Turing
     		 if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
     		 else printf("Unknown device type\n");
     		 break;
     	 case 8: // Ampere
     		 if (devProp.minor == 0) cores = mp * 64;
     		 else if (devProp.minor == 6) cores = mp * 128;
     		 else printf("Unknown device type\n");
     		 break;
     	 default:
     		 printf("Unknown device type\n");
     		 break;
    }
    return cores;
}
