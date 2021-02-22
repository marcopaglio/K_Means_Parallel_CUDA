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

__host__ void initialCentroids(unsigned int k, float* d_pointsCoordinates);

__host__ void updateClusters(unsigned int k, float* d_pointsCoordinates, unsigned int* clusterization, unsigned int* d_clusterSize, float* d_clusterSum);

__host__ bool checkStop(unsigned int k, const float* d_oldCentroidsCoordinates);


#endif /* KMEANS_H_ */
