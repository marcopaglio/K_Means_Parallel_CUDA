#include <iostream>
#include "Utils.h"
#include "KMeans.h"

using namespace std;

#define MAX_K 10

int main() {
	int k = 6;

	Image* img = load("/home/marco/eclipse-workspace/K_Means_Parallel_CUDA/src/Image/test.ppm");

	SetOfPoints data = pixelize(img);

	SetOfPoints* clusters = kMeans(k, data);

	saveRGBimage2D(k, clusters, "/home/marco/eclipse-workspace/K_Means_Parallel_CUDA/out/results/testOut.ppm", img);

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
