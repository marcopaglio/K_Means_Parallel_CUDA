/*
 * KMeans.h
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#ifndef KMEANS_H_
#define KMEANS_H_

#include "SetOfPoints.h"
#include "Point.h"

__host__ SetOfPoints* kMeans(unsigned int k, const SetOfPoints& data) noexcept(false);

__host__ Point* initialCentroids(unsigned int k, unsigned int dim, const SetOfPoints& data, float* centroidCoordinates);

__host__ void updateClusters(unsigned int k, const SetOfPoints& data/*, Point* centroids*/, unsigned int* clusterization, unsigned int* clusterSize, float* sum);

__host__ Point* updateCentroids(unsigned int k, unsigned int dim, unsigned int* clusterSize, float* sum, float* centroidCoordinates);

__host__ bool checkStop(unsigned int k, const Point* oldCentroids, const Point* newCentroids);


#endif /* KMEANS_H_ */
