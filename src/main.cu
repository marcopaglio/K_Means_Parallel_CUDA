#include <iostream>
#include "Utils.h"
#include "KMeans.h"
#include "Image.h"

#include <chrono>
#include <thread>

using namespace std;

#define MAX_K 10

int main() {
	int k = 4;

	Image* img = loadJPEG("/home/kevin/git/K_Means_Parallel_CUDA/src/Image/monarch.jpg");

	SetOfPoints data = pixelize(img);

	SetOfPoints* clusters = kMeans(k, data);

	savePNG(clusters, k, "/home/kevin/git/K_Means_Parallel_CUDA/out/results/testOut.png", img->width, img->height);

	for (int p = 0; p < Image_getWidth(img) * Image_getHeight(img); p++) {
		free(data.pointList[p].coordinates);
	}
	free(data.pointList);
	if (k != 1) {
		for (int c = 0; c < k; c++) {
			free(clusters[c].pointList);
		}
	}
	free(img);
	free(clusters);

    return 0;
}
