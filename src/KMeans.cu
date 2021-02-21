/*
 * KMeans.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "KMeans.h"
#include <cmath>
#include "Utils.h"

#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define TOLERANCE 0.005
#define BLOCK_SIZE 512
#define SMALL_BLOCK_SIZE 128

__device__ unsigned int d_dimension;
__device__ unsigned int d_newCentroidIndex;
__device__ float d_maxMinDistance;
__device__ unsigned int cudaLock;
extern __constant__ float c_centroidsCoordinates[];
extern const unsigned int channels;		//extern per prelevare var pubblica
static float* h_centroidsCoordinates;
static unsigned int h_dimension;
static unsigned int numPoints;

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
    Point* points = data.pointList;
    if(points == nullptr) {
        throw invalid_argument("Clusters can't be null");
    }
    numPoints = data.sizeList;
    if (numPoints < k) {
        throw length_error("There aren't enough points for k = " + to_string(k));
    }

    SetOfPoints* clusters = SetOfPoints_new((Point *) calloc(k, sizeof (Point)), k);
    if (k == 1) {
        setAttributes(&clusters[0], points, numPoints);
        return clusters;
    }

    h_dimension = data.pointList[0].dimension;
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_dimension, &h_dimension, sizeof(unsigned int)));

    float* d_pointsCoordinates; //TODO sposta su memoria globale
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pointsCoordinates, h_dimension * numPoints * sizeof(float)));
	for (unsigned int p = 0; p < numPoints; p++) {
		CUDA_CHECK_RETURN(cudaMemcpy((void*)&d_pointsCoordinates[p * h_dimension],
				(void*)data.pointList[p].coordinates, h_dimension * sizeof(float), cudaMemcpyHostToDevice));
	}

	CUDA_CHECK_RETURN(cudaGetSymbolAddress((void**)&h_centroidsCoordinates, c_centroidsCoordinates));
    assert(h_dimension == channels);
    initialCentroids(k, d_pointsCoordinates);
    float* d_oldCentroidsCoordinates;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_oldCentroidsCoordinates, k * h_dimension * sizeof(float)));

    unsigned int* clusterization = (unsigned int *) calloc(numPoints, sizeof(unsigned int));
    //unsigned int* clusterization;
    //CUDA_CHECK_RETURN(cudaMallocHost((void**)&clusterization, numPoints * sizeof(unsigned int), cudaHostAllocMapped));
    unsigned int* clusterSize = (unsigned int *) calloc(k, sizeof(unsigned int));
    float* clusterSum = (float *) calloc(k * h_dimension, sizeof(float));

    bool stop = false;
    while (!stop) {
		CUDA_CHECK_RETURN(cudaMemcpy((void*)d_oldCentroidsCoordinates, (void*)h_centroidsCoordinates, k * h_dimension * sizeof(float), cudaMemcpyDeviceToDevice));

		updateClusters(k, d_pointsCoordinates, clusterization, clusterSize, clusterSum);
        updateCentroids(k, clusterSize, clusterSum);

        if(checkStop(k, d_oldCentroidsCoordinates)) {
        	stop = true;
        }
    }

    free(clusterSum);

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
    free(clusterization);
    CUDA_CHECK_RETURN(cudaFree(d_pointsCoordinates));

    //CUDA_CHECK_RETURN(cudaFreeHost(clusterization));
    return clusters;
}

__global__ void checkStopKernel(unsigned int k, const float* d_oldCentroidsCoordinates, int* d_stop) {
	int coordinate = blockIdx.x * blockDim.x + threadIdx.x;
	if (coordinate < k * d_dimension) {
		if (abs(d_oldCentroidsCoordinates[coordinate] - c_centroidsCoordinates[coordinate]) > TOLERANCE) {
			atomicExch(d_stop, 1);
		}
	}
}

__host__ bool checkStop(unsigned int k, const float* d_oldCentroidsCoordinates) {
	int* d_stop;
	int h_stop;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_stop, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(d_stop, 0, sizeof(int)));

	static unsigned int gridSize = (k*h_dimension)/SMALL_BLOCK_SIZE + ((k * h_dimension) % SMALL_BLOCK_SIZE != 0);	// integer ceil

	checkStopKernel<<<gridSize, SMALL_BLOCK_SIZE>>>(k, d_oldCentroidsCoordinates, d_stop);
	cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy((void*)&h_stop, (void*)d_stop, sizeof(int), cudaMemcpyDeviceToHost));
    return h_stop ? false : true;
}

__host__ Point* updateCentroids(unsigned int k, unsigned int* clusterSize, float* clusterSum) { //TODO creare con kernel
    string fakeMeta;
    float* coordinates;

    Point* centroids = (Point*) calloc(k, sizeof(Point));
    for (unsigned int c = 0; c < k; c++) {
        coordinates = (float*) calloc(h_dimension, sizeof(float));
        for (unsigned int d = 0; d < h_dimension; d++) {
            coordinates[d] = clusterSum[c * h_dimension + d] / (float) clusterSize[c];
        }
        setAttributes(&(centroids[c]), coordinates, h_dimension, fakeMeta);
		CUDA_CHECK_RETURN(cudaMemcpy((void*)&h_centroidsCoordinates[c * h_dimension], (void*)coordinates, h_dimension * sizeof(float), cudaMemcpyHostToDevice));
    }

    return centroids;
}

__global__ void updateClustersKernel(unsigned int k, const float* d_pointsCoordinates, unsigned int maxIndex, unsigned int* d_clusterization, unsigned int* d_clusterSize, float* d_clusterSum) {
	// indexes
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < maxIndex) {
		float minDistance = INFINITY;
		unsigned int nearestCentroid;
		float distance;

		for (unsigned int c = 0; c < k; c++) {
			distance = getDistanceByCoordinates(&d_pointsCoordinates[index * d_dimension], &c_centroidsCoordinates[c * d_dimension], d_dimension);
			if (distance < minDistance) {
				minDistance = distance;
				nearestCentroid = c;
			}
		}
		d_clusterization[index] = nearestCentroid;
		atomicAdd(&d_clusterSize[nearestCentroid], 1);
		for (unsigned int d = 0; d < d_dimension; d++) {
			atomicAdd(&d_clusterSum[nearestCentroid * d_dimension + d], d_pointsCoordinates[index * d_dimension + d]);
		}
	}
}

__host__ void updateClusters(unsigned int k, float* d_pointsCoordinates, unsigned int* clusterization, unsigned int* clusterSize, float* clusterSum) {
	// reset arrays
	for (unsigned int c = 0; c < k; c++) {
        clusterSize[c] = 0;
        for (unsigned int d = 0; d < h_dimension; d++) {
        	clusterSum[c * h_dimension + d] = 0;
        }
    }

	unsigned int* d_clusterSize;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_clusterSize, k * sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_clusterSize, (void*)clusterSize, k * sizeof(unsigned int), cudaMemcpyHostToDevice));
	float* d_clusterSum;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_clusterSum, k * h_dimension * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_clusterSum, (void*)clusterSum, k * h_dimension * sizeof(float), cudaMemcpyHostToDevice));
	unsigned int* d_clusterization;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_clusterization, numPoints * sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_clusterization, (void*)clusterization, numPoints * sizeof(unsigned int), cudaMemcpyHostToDevice));


	// blocks-threads organization
	static unsigned int gridSize = numPoints/BLOCK_SIZE + (numPoints % BLOCK_SIZE != 0);			// integer ceil

	updateClustersKernel<<<gridSize, BLOCK_SIZE>>>(k, d_pointsCoordinates, numPoints, d_clusterization, d_clusterSize, d_clusterSum);
	cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy((void*)clusterSum, (void*)d_clusterSum, k * h_dimension * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_clusterSum));
    CUDA_CHECK_RETURN(cudaMemcpy((void*)clusterSize, (void*)d_clusterSize, k * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_clusterSize));
    CUDA_CHECK_RETURN(cudaMemcpy((void*)clusterization, (void*)d_clusterization, numPoints * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_clusterization));
}

__global__ void maxMinDistanceKernel(unsigned int i, const float* d_pointsCoordinates, unsigned int maxIndex) {
	__shared__ float ds_maxMinDistances[BLOCK_SIZE];
	__shared__ unsigned int ds_maxMinIndexes[BLOCK_SIZE];

	// indexes
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;

	// collaborative initialization
	ds_maxMinDistances[t] = -1;
	ds_maxMinIndexes[t] = index;

	// points processing
	if (index < maxIndex) {
		float minDistance = INFINITY;
		float distance;
		for (unsigned int j = 0; j < i; j++) {
			distance = getDistanceByCoordinates(&d_pointsCoordinates[index * d_dimension], &c_centroidsCoordinates[j * d_dimension], d_dimension);
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
    CUDA_CHECK_RETURN(cudaMemcpy((void*)&h_centroidsCoordinates[0 * h_dimension], (void*)&d_pointsCoordinates[firstIndex * h_dimension], h_dimension * sizeof(float), cudaMemcpyHostToDevice));

    // blocks-threads organization
    unsigned int gridSize = numPoints/BLOCK_SIZE + (numPoints % BLOCK_SIZE != 0);			// integer ceil

    // kernel call iteration
    unsigned int h_newCentroidIndex;
    float h_maxMinDistance;

    for (unsigned int i = 1; i < k; i++) {
        h_maxMinDistance = 0;
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_maxMinDistance, &h_maxMinDistance, sizeof(float)));

        maxMinDistanceKernel<<<gridSize, BLOCK_SIZE>>>(i, d_pointsCoordinates, numPoints);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&h_newCentroidIndex, d_newCentroidIndex, sizeof(unsigned int)));
        CUDA_CHECK_RETURN(cudaMemcpy((void*)&h_centroidsCoordinates[i * h_dimension], (void*)&d_pointsCoordinates[h_newCentroidIndex * h_dimension], h_dimension * sizeof(float), cudaMemcpyHostToDevice));
    }
}
