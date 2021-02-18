#include <iostream>
#include "Utils.h"
#include "KMeans.h"
#include "Image.h"

using namespace std;

#define MAX_K 10
#define k 3
extern const unsigned int channels = 3;			//extern rende pubblica var const
__constant__ float c_centroidsCoordinates[k * channels];

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

int main() {

	Image* img = loadJPEG("/home/marco/eclipse-workspace/K_Means_Parallel_CUDA/src/Image/mountain.jpg");

	SetOfPoints data = pixelize(img);

	SetOfPoints* clusters = kMeans(k, data);

	savePNG(clusters, k, "/home/marco/eclipse-workspace/K_Means_Parallel_CUDA/out/results/mountainOut.png", img->width, img->height);

	for (int p = 0; p < Image_getWidth(img) * Image_getHeight(img); p++) {
		//free(data.pointList[p].coordinates);
		CUDA_CHECK_RETURN(cudaFreeHost(data.pointList[p].coordinates));
	}
	//free(data.pointList);
	cudaFreeHost(data.pointList);
	if (k != 1) {
		for (int c = 0; c < k; c++) {
			free(clusters[c].pointList);
		}
	}
	free(img);
	free(clusters);

    return 0;
}
