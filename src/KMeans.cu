/*
 * KMeans.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "KMeans.h"
#include <cmath>

#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define TOLERANCE 0.005
#define BLOCK_SIZE 384
#define SMALL_BLOCK_SIZE 128

extern const unsigned int channels;
static unsigned int dimension;
static unsigned int numPoints;
static float* h_centroidsCoordinates;

__constant__ unsigned int c_dimension;
__constant__ unsigned int c_numPoints;
extern __constant__ float c_centroidsCoordinates[];

__device__ unsigned int d_newCentroidIndex;
__device__ float d_maxMinDistance;
__device__ unsigned int cudaLock;
extern __device__ unsigned int g_clusterSize[];
extern __device__ float g_clusterSum[];

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

__host__ SetOfPoints* kMeans(unsigned int k, const SetOfPoints& data) noexcept(false) {
	// extract points
    Point* points = data.pointList;
    if(points == nullptr) {
        throw invalid_argument("Clusters can't be null");
    }

    // extract size
    numPoints = data.sizeList;
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_numPoints, &numPoints, sizeof(unsigned int)));
    if (numPoints < k) {
        throw length_error("There aren't enough points for k = " + to_string(k));
    }

    // extract dimension
    dimension = data.pointList[0].dimension;
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_dimension, &dimension, sizeof(unsigned int)));
    assert(dimension == channels);

    // initialize linear array on device batching the memory transfers and pinning memory
	float* pointsCoordinates;
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&(pointsCoordinates), dimension * numPoints * sizeof(float), cudaHostAllocMapped));
	for (int p = 0; p < numPoints; p++){
		memcpy((void*)&pointsCoordinates[p*dimension], (void*)points[p].coordinates, dimension*sizeof(float));
	}
	float* d_pointsCoordinates;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pointsCoordinates, dimension * numPoints * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_pointsCoordinates, (void*)pointsCoordinates, dimension * numPoints *sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaFreeHost(pointsCoordinates));

	// initialize centroids
	CUDA_CHECK_RETURN(cudaGetSymbolAddress((void**)&h_centroidsCoordinates, c_centroidsCoordinates));
    initialCentroids(k, d_pointsCoordinates);

    unsigned int* clusterization;
    CUDA_CHECK_RETURN(cudaMallocManaged(&clusterization, numPoints*sizeof(unsigned int)));
    cudaMemPrefetchAsync(clusterization, numPoints*sizeof(unsigned int), -1, NULL);			// for post-Maxwell architectures
//	unsigned int* d_clusterization;
//	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_clusterization, numPoints * sizeof(unsigned int)));
	unsigned int* d_clusterSize;
	CUDA_CHECK_RETURN(cudaGetSymbolAddress((void**)&d_clusterSize, g_clusterSize));
	float* d_clusterSum;
	CUDA_CHECK_RETURN(cudaGetSymbolAddress((void**)&d_clusterSum, g_clusterSum));

    bool stop = false;
    while (!stop) {
		updateClusters(k, d_pointsCoordinates, /*d_clusterization*/clusterization, d_clusterSize, d_clusterSum);

        stop = checkStop(k);
        CUDA_CHECK_RETURN(cudaMemcpy((void*)h_centroidsCoordinates, (void*)d_clusterSum, k * dimension * sizeof(float), cudaMemcpyDeviceToDevice));
    }

//  unsigned int* clusterization = (unsigned int *) calloc(numPoints, sizeof(unsigned int));
//  CUDA_CHECK_RETURN(cudaMemcpy((void*)clusterization, (void*)d_clusterization, numPoints * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int* clusterSize = (unsigned int *) calloc(k, sizeof(unsigned int));
    CUDA_CHECK_RETURN(cudaMemcpy((void*)clusterSize, (void*)d_clusterSize, k * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // alloc clusters
    SetOfPoints* clusters = SetOfPoints_new((Point *) calloc(k, sizeof (Point)), k);
    unsigned int clusterIndex [k];
	for (unsigned int c = 0; c < k; c++) {
		setAttributes(&(clusters[c]), (Point *) calloc(clusterSize[c], sizeof(Point)), clusterSize[c]);
		clusterIndex[c] = 0;
	}
	for (unsigned int p = 0; p < numPoints; p++) {
		insertPoint(&(clusters[clusterization[p]]), points[p], clusterIndex[clusterization[p]]);
		clusterIndex[clusterization[p]]++;
	}

    free(clusterSize);
    CUDA_CHECK_RETURN(cudaFree(clusterization));
//  free(clusterization);
//  CUDA_CHECK_RETURN(cudaFree(d_clusterization));
    CUDA_CHECK_RETURN(cudaFree(d_pointsCoordinates));

    return clusters;
}

__global__ void checkStopKernel(unsigned int k, int* d_stop) {
	// indexes
	int coordinate = blockIdx.x * blockDim.x + threadIdx.x;

	// private copies
	int dimension = c_dimension;

	// main job
	if (coordinate < k * dimension) {
		float actualCoordinate = g_clusterSum[coordinate] / (float) g_clusterSize[coordinate / dimension];
		g_clusterSum[coordinate] = actualCoordinate;
		if (abs(c_centroidsCoordinates[coordinate] - actualCoordinate) > TOLERANCE) {
			atomicExch(d_stop, 1);
		}
	}
}

__host__ bool checkStop(unsigned int k) {
    int* stop;
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&stop, sizeof(int)));

	static unsigned int gridSize = (k*dimension)/SMALL_BLOCK_SIZE + ((k * dimension) % SMALL_BLOCK_SIZE != 0);	// integer ceil

	checkStopKernel<<<gridSize, SMALL_BLOCK_SIZE>>>(k, stop);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	bool response = *stop ? false : true;
	CUDA_CHECK_RETURN(cudaFree(stop));

	return response;
}

__global__ void updateClustersKernel(unsigned int k, const float* d_pointsCoordinates, unsigned int* d_clusterization) {
	// indexes
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// private copies
	int dimension = c_dimension;
	int numPoints = c_numPoints;

	// main job
	if (index < numPoints) {
		float minDistance = INFINITY;
		unsigned int nearestCentroid;
		float distance;

		for (unsigned int c = 0; c < k; c++) {
			distance = getDistanceByCoordinates(&d_pointsCoordinates[index * dimension], &c_centroidsCoordinates[c * dimension], dimension);
			if (distance < minDistance) {
				minDistance = distance;
				nearestCentroid = c;
			}
		}
		d_clusterization[index] = nearestCentroid;
		atomicAdd(&g_clusterSize[nearestCentroid], 1);
		for (unsigned int d = 0; d < dimension; d++) {
			atomicAdd(&g_clusterSum[nearestCentroid * dimension + d], d_pointsCoordinates[index * dimension + d]);
		}
	}
}

__host__ void updateClusters(unsigned int k, float* d_pointsCoordinates, unsigned int* d_clusterization, unsigned int* d_clusterSize, float* d_clusterSum) {
	CUDA_CHECK_RETURN(cudaMemset((void*)d_clusterSize, 0, k * sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemset((void*)d_clusterSum, 0, k * dimension * sizeof(float)));

	// blocks-threads organization
	static unsigned int gridSize = numPoints/BLOCK_SIZE + (numPoints % BLOCK_SIZE != 0);			// integer ceil

	updateClustersKernel<<<gridSize, BLOCK_SIZE, k*dimension>>>(k, d_pointsCoordinates, d_clusterization);
	cudaDeviceSynchronize();
}

__global__ void maxMinDistanceKernel(unsigned int i, const float* d_pointsCoordinates) {
	__shared__ float ds_maxMinDistances[BLOCK_SIZE];
	__shared__ unsigned int ds_maxMinIndexes[BLOCK_SIZE];

	// private copies
	int dimension = c_dimension;
	int numPoints = c_numPoints;

	// indexes
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;

	// collaborative initialization
	ds_maxMinDistances[t] = -1;
	ds_maxMinIndexes[t] = index;

	// points processing
	if (index < numPoints) {
		float minDistance = INFINITY;
		float distance;
		for (unsigned int j = 0; j < i; j++) {
			distance = getDistanceByCoordinates(&d_pointsCoordinates[index * dimension], &c_centroidsCoordinates[j * dimension], dimension);
			if (distance < minDistance) {
				minDistance = distance;
			}
		}

		ds_maxMinDistances[t] = minDistance;
	}

	// comparing reduction
	for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
		__syncthreads();

		if (t >= stride) {
			return; 		// this improves kernel's performance
		} else {
			if (ds_maxMinDistances[t] < ds_maxMinDistances[t + stride]) {
				ds_maxMinDistances[t] = ds_maxMinDistances[t + stride];
				ds_maxMinIndexes[t] = ds_maxMinIndexes[t + stride];
			}
			if (stride == 1) {
				bool blocked = true;
				while (blocked) {
					if (0 == atomicCAS(&cudaLock, 0, 1)) {
						if (d_maxMinDistance < ds_maxMinDistances[0]) {
							d_maxMinDistance = ds_maxMinDistances[0];
							d_newCentroidIndex = ds_maxMinIndexes[0];
						}
						atomicExch(&cudaLock, 0);
						blocked = false;
					}
				}
			}
		}
	}
}

__host__ void initialCentroids(unsigned int k, float* d_pointsCoordinates) {
	// first centroid
    int firstIndex = 0;
    CUDA_CHECK_RETURN(cudaMemcpy((void*)&h_centroidsCoordinates[0 * dimension], (void*)&d_pointsCoordinates[firstIndex * dimension], dimension * sizeof(float), cudaMemcpyHostToDevice));

    // blocks-threads organization
    unsigned int gridSize = numPoints/BLOCK_SIZE + (numPoints % BLOCK_SIZE != 0);			// integer ceil

    // kernel call iteration
    unsigned int h_newCentroidIndex;
    float h_maxMinDistance;

    for (unsigned int i = 1; i < k; i++) {
        h_maxMinDistance = 0;
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_maxMinDistance, &h_maxMinDistance, sizeof(float)));

        maxMinDistanceKernel<<<gridSize, BLOCK_SIZE>>>(i, d_pointsCoordinates);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&h_newCentroidIndex, d_newCentroidIndex, sizeof(unsigned int)));
        CUDA_CHECK_RETURN(cudaMemcpy((void*)&h_centroidsCoordinates[i * dimension], (void*)&d_pointsCoordinates[h_newCentroidIndex * dimension], dimension * sizeof(float), cudaMemcpyHostToDevice));
    }
}
