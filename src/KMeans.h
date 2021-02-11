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

__host__ SetOfPoints* kMeans(int k, const SetOfPoints& data) noexcept(false);

__host__ Point* initialCentroids(int k, const SetOfPoints& data);

void updateClusters(int k, Point* centroids, const SetOfPoints& data, int* clusterization, int* clusterSize);

Point* updateCentroids(int k, const SetOfPoints& data, const int* clusterization);

bool checkStop(int k, const Point* oldCentroids, const Point* newCentroids);

#endif /* KMEANS_H_ */
