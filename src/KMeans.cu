/*
 * KMeans.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "KMeans.h"
#include "Point.h"
#include <cmath>

#include <cstdio>
#include <cassert>
#include <iostream>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define TOLERANCE 0.005
__device__ unsigned int d_newCentroidIndex;
__device__ float d_maxMinDistance;
__device__ unsigned int cudaLock;

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

__host__ SetOfPoints* kMeans(unsigned int k, const SetOfPoints& data) noexcept(false) {
    Point* pointList = data.pointList;
    if(pointList == nullptr) {
        throw invalid_argument("Clusters can't be null");
    }
    int numPoints = data.sizeList;
    if (numPoints < k) {
        throw length_error("There aren't enough points for k = " + to_string(k));
    }

    SetOfPoints* clusters = SetOfPoints_new((Point *) calloc(k, sizeof (Point)), k);
    if (k == 1) {
        setAttributes(&(clusters[0]), pointList, numPoints);
        return clusters;
    }

    Point* centroids = initialCentroids(k, data);
    unsigned int* clusterization = (unsigned int *) calloc(numPoints, sizeof(unsigned int));
    Point* oldCentroids;
    unsigned int* clusterSize = (unsigned int *) calloc(k, sizeof(unsigned int));
    unsigned int dim = data.pointList[0].dimension;
    float* sum = (float *) calloc(k * dim, sizeof(float));

    bool stop = false;
    while (!stop) {
        updateClusters(k, data, centroids, clusterization, clusterSize, sum);
        oldCentroids = centroids;
        centroids = updateCentroids(k, dim, clusterSize, sum);

        if(checkStop(k, oldCentroids, centroids)) {
            stop = true;
        }

        for (int z = 0; z < k; z++) {
            free(oldCentroids[z].coordinates);
        }
        free(oldCentroids);
    }
    for (int z = 0; z < k; z++) {
        free(centroids[z].coordinates);
    }
    free(centroids);
    free(sum);

    unsigned int clusterIndex [k];
	for (unsigned int c = 0; c < k; c++) {
		setAttributes(&(clusters[c]), (Point *) calloc(clusterSize[c], sizeof(Point)), clusterSize[c]);
		clusterIndex[c] = 0;
	}
	for (unsigned int p = 0; p < numPoints; p++) {
		insertPoint(&(clusters[clusterization[p]]), pointList[p], clusterIndex[clusterization[p]]);
		clusterIndex[clusterization[p]]++;
	}

    free(clusterSize);
    free(clusterization);
    return clusters;
}

__host__ bool checkStop(unsigned int k, const Point* oldCentroids, const Point* newCentroids) {
    unsigned int dimension = oldCentroids[0].dimension;
    for (unsigned int c = 0; c < k; c++) {
        float* oldCoordinates = oldCentroids[c].coordinates;
        float* newCoordinates = newCentroids[c].coordinates;
        for (unsigned int d = 0; d < dimension; d++) {
            if (abs(oldCoordinates[d] - newCoordinates[d]) > TOLERANCE) {
                return false;
            }
        }
    }
    return true;
}

__host__ Point* updateCentroids(unsigned int k, unsigned int dim, unsigned int* clusterSize, float* sum) {
    string fakeMeta;

    Point* centroids = (Point*) calloc(k, sizeof(Point));
    for (unsigned int c = 0; c < k; c++) {
        float* coordinates = (float*) calloc(dim, sizeof(float));
        for (unsigned int d = 0; d < dim; d++) {
            coordinates[d] = sum[c * dim + d] / (float) clusterSize[c];
        }
        setAttributes(&(centroids[c]), coordinates, dim, fakeMeta);
    }

    return centroids;
}

__global__ void updateClustersKernel(unsigned int k, Point* d_pointList, unsigned int size, unsigned int from, Point* c_centroids, unsigned int* d_clusterization, unsigned int* d_clusterSize, float* d_sum) {
	unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;

	if (p < size) {
		unsigned int dim = d_pointList[0].dimension;
		float minDistance = INFINITY;
        unsigned int nearestCentroid;
        float distance;

        for (unsigned int c = 0; c < k; c++) {
            distance = getDistance(d_pointList[p], c_centroids[c]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCentroid = c;
            }
        }
        d_clusterization[from + p] = nearestCentroid;
        atomicAdd(&(d_clusterSize[nearestCentroid]), 1);
        for (unsigned int d = 0; d < dim; d++) {
        	atomicAdd(&(d_sum[nearestCentroid * dim + d]), getCoordinateAt(d_pointList[p], d));
        }
	}
}

__host__ void updateClusters(unsigned int k, const SetOfPoints& data, Point* centroids, unsigned int* clusterization, unsigned int* clusterSize, float* sum) {
    unsigned int numPoints = data.sizeList;
	unsigned int dim = data.pointList[0].dimension;
	for (unsigned int c = 0; c < k; c++) {
        clusterSize[c] = 0;
        for (unsigned int d = 0; d < dim; d++) {
        	sum[c * dim + d] = 0;
        }
    }

	unsigned int* d_clusterSize;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_clusterSize, k * sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_clusterSize, (void*)clusterSize, k * sizeof(unsigned int), cudaMemcpyHostToDevice));
	unsigned int* d_clusterization;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_clusterization, numPoints * sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_clusterization, (void*)clusterization, numPoints * sizeof(unsigned int), cudaMemcpyHostToDevice));
	float* d_sum;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_sum, k * dim * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_sum, (void*)sum, k * dim * sizeof(float), cudaMemcpyHostToDevice));
	float* d_copyCoordinates;
	Point* c_centroids;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_centroids, k * sizeof(Point)));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)c_centroids, (void*)centroids, k * sizeof(Point), cudaMemcpyHostToDevice));
	for (unsigned int c = 0; c < k; c++) {
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_copyCoordinates, dim * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMemcpy((void*)d_copyCoordinates,
				(void*)centroids[c].coordinates, dim * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy((void*)&((c_centroids + c)->coordinates), (void*)&d_copyCoordinates,
				sizeof((c_centroids + c)->coordinates), cudaMemcpyHostToDevice));
	}

	cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop, 0);

	unsigned int blockSize = 128;
	unsigned int gridSizeNecessary = ceil(data.sizeList / (float) blockSize);
    unsigned int gridSize = dev_prop.maxGridSize[0];

    Point* d_pointList;
    unsigned int from = 0;
    unsigned int size;
    unsigned int p;

    while (gridSizeNecessary > 0) {
		if (gridSizeNecessary > dev_prop.maxGridSize[0]) { //maxGridSize = 2 miliardi, sicuramente troppo, vedi FIXME successivo
			size = gridSize * blockSize;
		} else {
			size = data.sizeList - from;
			gridSize = gridSizeNecessary;
		}
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pointList, size * sizeof(Point))); //FIXME returned out of memory because too much great
		CUDA_CHECK_RETURN(cudaMemcpy((void*)d_pointList, (void*)&data.pointList[from],
				size * sizeof(Point), cudaMemcpyHostToDevice));
		for (p = 0; p < size; p++) {
			CUDA_CHECK_RETURN(cudaMalloc((void**)&d_copyCoordinates, dim * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMemcpy((void*)d_copyCoordinates,
					(void*)data.pointList[from + p].coordinates, dim * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy((void*)&((d_pointList + p)->coordinates),
					(void*)&d_copyCoordinates, sizeof((d_pointList + p)->coordinates), cudaMemcpyHostToDevice));
		}

		//CALL kernel
		updateClustersKernel<<<gridSize, blockSize>>>(k, d_pointList, size, from, c_centroids, d_clusterization, d_clusterSize, d_sum);

		from += size;
		gridSizeNecessary -= gridSize;
		cudaDeviceSynchronize();

		for (p = 0; p < size; p++) {
			CUDA_CHECK_RETURN(cudaMemcpy((void*)&d_copyCoordinates, (void*)&((d_pointList + p)->coordinates),
							sizeof((d_pointList + p)->coordinates), cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaFree(d_copyCoordinates));
		}
		CUDA_CHECK_RETURN(cudaFree(d_pointList));
	}

    CUDA_CHECK_RETURN(cudaMemcpy((void*)sum, (void*)d_sum, k * dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_sum));
    CUDA_CHECK_RETURN(cudaMemcpy((void*)clusterSize, (void*)d_clusterSize, k * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_clusterSize));
    CUDA_CHECK_RETURN(cudaMemcpy((void*)clusterization, (void*)d_clusterization, numPoints * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(d_clusterization));
    for (unsigned int c = 0; c < k; c++) {
		CUDA_CHECK_RETURN(cudaMemcpy((void*)&d_copyCoordinates, (void*)&((c_centroids + c)->coordinates),
				sizeof((c_centroids + c)->coordinates), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaFree(d_copyCoordinates));
	}
	CUDA_CHECK_RETURN(cudaFree(c_centroids));
}

__global__ void maxMinDistanceKernel(unsigned int i, Point* d_pointList, unsigned int size, unsigned int from, Point* c_centroids) {
	__shared__ float ds_maxMinDistances[128];
	__shared__ unsigned int ds_maxMinIndexes[128];

	unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;

	if (p % 1000 == 0) {
		printf("K = %d, thread %d utilizza r: %f, g: %f, b: %f.\n\n", i, p, d_pointList[p].coordinates[0], d_pointList[p].coordinates[1], d_pointList[p].coordinates[2]);
	}
	if (p < size) {
		float minDistance = INFINITY;
		float distance;
		for (unsigned int j = 0; j < i; j++) {
			distance = getDistance(d_pointList[p], c_centroids[j]);
			if (distance < minDistance) {
				minDistance = distance;
			}
		}

		ds_maxMinDistances[t] = minDistance;
		ds_maxMinIndexes[t] = from + p;
	} else {
		ds_maxMinDistances[t] = 0;			//0 will never be caught
		ds_maxMinIndexes[t] = from + size;	//from+size is out of range
	}

	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();

		if (t >= stride) {
			return; //TESTED with k=10 less than if(t < stride) {} (TOT: 82s vs 83s)
		} else {
		//if(t < stride){
			if (ds_maxMinDistances[t] < ds_maxMinDistances[t + stride]) {
				ds_maxMinDistances[t] = ds_maxMinDistances[t + stride];
				ds_maxMinIndexes[t] = ds_maxMinIndexes[t + stride];
			}
			if (stride == 1) {
				bool blocked = true;
				while (blocked) {
					if (0 == atomicCAS(&cudaLock, 0, 1)) {
						if (d_maxMinDistance < ds_maxMinDistances[0]) {
							atomicExch(&d_maxMinDistance, ds_maxMinDistances[0]);
							atomicExch(&d_newCentroidIndex, ds_maxMinIndexes[0]);
						}
						atomicExch(&cudaLock, 0);
						blocked = false;
					}
				}
			}
		}
	}
}

__host__ Point* initialCentroids(unsigned int k, const SetOfPoints& data) {
    if (data.sizeList == k) {
        return data.pointList;
    }
    Point* centroids = (Point*) calloc(k, sizeof(Point));
	Point* c_centroids;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_centroids, k * sizeof(Point)));

	unsigned int dim = data.pointList[0].dimension;
	float* d_copyCoordinates;

    int firstIndex = 0;
    copyPoint(&(centroids[0]), &(data.pointList[firstIndex]));

    /**** NEXT CODE IS USED TO VERIFY INIT IS DONE
    std::cout << "Centroide 0-esimo" << std::endl;
    std::cout << "dim: " << centroids[0].dimension << std::endl;
    std::cout << "meta " << centroids[0].metadata << std::endl;
    for(int i=0; i<dim; i++) {
		std::cout << i << " " << centroids[0].coordinates[i] << std::endl;
	}
	*****/

    //Every deep copy in device memory consists in 4 steps:
    //1) init struct in device memory
    CUDA_CHECK_RETURN(cudaMemcpy((void*)&(c_centroids[0]),
        				(void*)&centroids[0], sizeof(Point), cudaMemcpyHostToDevice));
    //2) allocate array in device memory
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_copyCoordinates, dim * sizeof(float)));
    //3) init array in device memory
    CUDA_CHECK_RETURN(cudaMemcpy((void*)d_copyCoordinates,
    		(void*)centroids[0].coordinates, dim * sizeof(float), cudaMemcpyHostToDevice));

    /**** NEXT CODE IS USED TO VERIFY COPY IS DONE
    memset(centroids[0].coordinates, 0, dim * sizeof(float));
    cudaMemcpy(&centroids[0], &c_centroids[0], sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids[0].coordinates, d_copyCoordinates, dim * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "dim " << centroids[0].dimension << std::endl;
    std::cout << "meta " << centroids[0].metadata << std::endl;
    for(int i=0; i<dim; i++) {
        std::cout << i << " " << centroids[0].coordinates[i] << std::endl;
    }
    *****/

    //4) move device array pointer to device struct field
    CUDA_CHECK_RETURN(cudaMemcpy((void*)&((c_centroids + 0)->coordinates), (void*)&d_copyCoordinates,
    		sizeof((c_centroids + 0)->coordinates), cudaMemcpyHostToDevice));

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0); //device = GPU ?? Sì

    unsigned int blockSize = 128;
    unsigned int gridSizeNecessary;

    unsigned int h_newCentroidIndex;
    float h_maxMinDistance;

    Point* d_pointList;
    unsigned int from;
    unsigned int size;
    unsigned int gridSize = dev_prop.maxGridSize[0];
    unsigned int p;

    for (unsigned int i = 1; i < k; i++) {
        h_maxMinDistance = 0;
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_maxMinDistance, &h_maxMinDistance, sizeof(float)));

        /**** NEXT CODE IS USED TO VERIFY COPY IS DONE
        CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&h_maxMinDistance, d_maxMinDistance, sizeof(float)));
        std::cout << "init d_maxMinDistance at " << h_maxMinDistance << std::endl;
        ****/

        from = 0;
        gridSizeNecessary = ceil(data.sizeList / (float) blockSize); //need cast to float in order to have right value

        while (gridSizeNecessary > 0) {
        	if (gridSizeNecessary > dev_prop.maxGridSize[0]) { //maxGridSize = 2 miliardi, sicuramente troppo, vedi FIXME successivo
        		size = gridSize * blockSize;
        	} else {
        		size = data.sizeList - from;
        		gridSize = gridSizeNecessary;
        	}
        	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pointList, size * sizeof(Point))); //FIXME returned out of memory because too much great
        	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_pointList, (void*)&data.pointList[from],
        			size * sizeof(Point), cudaMemcpyHostToDevice));
        	for (p = 0; p < size; p++) {
        		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_copyCoordinates, dim * sizeof(float)));
        		CUDA_CHECK_RETURN(cudaMemcpy((void*)d_copyCoordinates,
        				(void*)data.pointList[from + p].coordinates, dim * sizeof(float), cudaMemcpyHostToDevice));
        		CUDA_CHECK_RETURN(cudaMemcpy((void*)&((d_pointList + p)->coordinates),
        				(void*)&d_copyCoordinates, sizeof((d_pointList + p)->coordinates), cudaMemcpyHostToDevice));

				/**** NEXT CODE IS USED TO VERIFY COPY IS DONE
				if (p % 1000 == 0) {
					std::cout << "Punto " << p << "-esimo: " << std::endl;
					for(int i=0; i<dim; i++) {
						std::cout << i << " " << data.pointList[from + p].coordinates[i] << std::endl;
					}
					memset(data.pointList[from + p].coordinates, 0, dim * sizeof(float));
					cudaMemcpy(data.pointList[from + p].coordinates, d_copyCoordinates, dim * sizeof(float), cudaMemcpyDeviceToHost);
					for(int i=0; i<dim; i++) {
						std::cout << i << " " << data.pointList[from + p].coordinates[i] << std::endl;
					}
				}
				****/
        	}

        	//CALL kernel
        	maxMinDistanceKernel<<<gridSize, blockSize>>>(i, d_pointList, size, from, c_centroids);

        	from += size;
        	gridSizeNecessary -= gridSize;
        	cudaDeviceSynchronize();

        	/**** NEXT CODE IS USED TO VERIFY IF KERNEL HAS LAUNCHED ERRORS
        	std::string error = cudaGetErrorString(cudaPeekAtLastError());
        	std::cout << error << std::endl;
        	error = cudaGetErrorString(cudaThreadSynchronize());
        	std::cout << error << std::endl;
        	****/

        	for (p = 0; p < size; p++) {
				//access to device pointer is not possible, so...
				//1) copy pointer of device pointer in pointer of host-defined device pointer
				CUDA_CHECK_RETURN(cudaMemcpy((void*)&d_copyCoordinates, (void*)&((d_pointList + p)->coordinates),
								sizeof((d_pointList + p)->coordinates), cudaMemcpyDeviceToHost));
				//2) free host defined device pointer
				CUDA_CHECK_RETURN(cudaFree(d_copyCoordinates));
			}
			// This doesn't need to copy because is a host-defined device pointer
			CUDA_CHECK_RETURN(cudaFree(d_pointList));
        }

        CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&h_newCentroidIndex, d_newCentroidIndex, sizeof(unsigned int)));
        copyPoint(&(centroids[i]), &(data.pointList[h_newCentroidIndex]));

        /**** NEXT CODE IS USED TO VERIFY INIT IS DONE
        std::cout << "Centroide " << i << "-esimo" << std::endl;
        std::cout << "dim: " << centroids[i].dimension << std::endl;
        std::cout << "meta " << centroids[i].metadata << std::endl;
        for(int j=0; j<dim; j++) {
    		std::cout << j << " " << centroids[i].coordinates[j] << std::endl;
    	}
    	****/

        CUDA_CHECK_RETURN(cudaMemcpy((void*)&c_centroids[i],
        		(void*)&centroids[i], sizeof(Point), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMalloc((void**)&d_copyCoordinates, dim * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy((void*)d_copyCoordinates,
        		(void*)centroids[i].coordinates, dim * sizeof(float), cudaMemcpyHostToDevice));

        /**** NEXT CODE IS USED TO VERIFY COPY IS DONE
        memset(centroids[i].coordinates, 0, dim * sizeof(float));
        cudaMemcpy(&centroids[i], &c_centroids[i], sizeof(Point), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids[i].coordinates, d_copyCoordinates, dim * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "dim " << centroids[i].dimension << std::endl;
        std::cout << "meta " << centroids[i].metadata << std::endl;
        for(int j=0; j<dim; j++) {
            std::cout << j << " " << centroids[i].coordinates[j] << std::endl;
        }
        ****/

        CUDA_CHECK_RETURN(cudaMemcpy((void*)&((c_centroids + i)->coordinates), (void*)&d_copyCoordinates,
        		sizeof((c_centroids + i)->coordinates), cudaMemcpyHostToDevice));
    }

    for (unsigned int c = 0; c < k; c++) {
    	CUDA_CHECK_RETURN(cudaMemcpy((void*)&d_copyCoordinates, (void*)&((c_centroids + c)->coordinates),
    			sizeof((c_centroids + c)->coordinates), cudaMemcpyDeviceToHost));
    	CUDA_CHECK_RETURN(cudaFree(d_copyCoordinates));
    }
    CUDA_CHECK_RETURN(cudaFree(c_centroids));

    return centroids;
}
