#include <iostream>
#include <chrono>

#include "Utils.h"
#include "KMeans.h"
#include "Image.h"

using namespace std;
using namespace std::chrono;

#define MIN_K 8
#define MAX_K 8
#define STEP_K 2
extern const unsigned int channels = 3;
__constant__ float c_centroidsCoordinates[MAX_K * channels];
__device__ unsigned int g_clusterSize[MAX_K];
__device__ float g_clusterSum[MAX_K * channels];

int main() {
	string filename = "6K-1";
	string pathIn = "/home/kevin/git/K_Means_Parallel_CUDA/src/Image/" + filename + ".jpg";

	Image* img = loadJPEG(pathIn.c_str());
	SetOfPoints data = pixelize(img);

	string pathOut;
	for (unsigned int k = MIN_K; k <= MAX_K; k += STEP_K) {

		auto start = system_clock::now();

		SetOfPoints* clusters = kMeans(k, data);

		auto end = system_clock::now();

		auto duration = duration_cast<milliseconds>(end - start);

		cout << "K = " << k << ": " << duration.count() << "ms" << endl;

		pathOut = "/home/kevin/git/K_Means_Parallel_CUDA/out/results/" + filename + to_string(k) + "Out.png";
		savePNG(clusters, k, pathOut.c_str(), img->width, img->height);

		for (int c = 0; c < k; c++) {
			free(clusters[c].pointList);
		}
		free(clusters);
	}

	for (int p = 0; p < Image_getWidth(img) * Image_getHeight(img); p++) {
		Point_delete(&data.pointList[p]);
	}
	free(data.pointList);
	Image_delete(img);

	return 0;
}
