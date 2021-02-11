/*
 * KMeans.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "KMeans.h"
#include <cmath>

#define TOLERANCE 0.005

SetOfPoints* kMeans(int k, const SetOfPoints& data) noexcept(false) {
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
    int* clusterization = (int *) calloc(numPoints, sizeof(int));
    Point* oldCentroids;
    int* clusterSize = (int *) calloc(k, sizeof(int));

    bool stop = false;
    while (!stop) {
        updateClusters(k, centroids, data, clusterization, clusterSize);
        oldCentroids = centroids;
        centroids = updateCentroids(k, data, clusterization);

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

    int clusterIndex [k];
    for (int i = 0; i < k; i++) {
        setAttributes(&(clusters[i]), (Point *) calloc(clusterSize[i], sizeof(Point)), clusterSize[i]);
        clusterIndex[i] = 0;
    }
    for (int i = 0; i < numPoints; i++) {
        insertPoint(&(clusters[clusterization[i]]), pointList[i], clusterIndex[clusterization[i]]);
        clusterIndex[clusterization[i]]++;
    }

    free(clusterSize);
    free(clusterization);
    return clusters;
}

bool checkStop(int k, const Point* oldCentroids, const Point* newCentroids) {
    int dimension = oldCentroids[0].dimension;
    for (int c = 0; c < k; c++) {
        float* oldCoordinates = oldCentroids[c].coordinates;
        float* newCoordinates = newCentroids[c].coordinates;
        for (int i = 0; i < dimension; i++) {
            if (abs(oldCoordinates[i] - newCoordinates[i]) > TOLERANCE) {
                return false;
            }
        }
    }
    return true;
}

Point* updateCentroids(int k, const SetOfPoints& data, const int* clusterization) {
    int dimension = data.pointList[0].dimension;

    // init data structures used
    float sum[k][dimension];
    int clustersSize [k];
    for (int c = 0; c < k; c++) {
        clustersSize[c] = 0;
        for (int d = 0; d < dimension; d++) {
            sum[c][d] = 0.0;
        }
    }

    for (int p = 0; p < data.sizeList; p++) {
        for (int d = 0; d < dimension; d++) {
            sum[clusterization[p]][d] += getCoordinateAt(data.pointList[p], d);
        }
        clustersSize[clusterization[p]]++;
    }

    Point* centroids = (Point*) calloc(k, sizeof(Point));
    for (int c = 0; c < k; c++) {
        float* coordinates = (float*) calloc(dimension, sizeof(float));
        for (int d = 0; d < dimension; d++) {
            coordinates[d] = sum[c][d] / (float) clustersSize[c];
        }

        string fakeMeta;
        setAttributes(&(centroids[c]), coordinates, dimension, fakeMeta);
    }

    return centroids;
}

void updateClusters(int k, Point* centroids, const SetOfPoints& data, int* clusterization, int* clusterSize) {
    for (int i = 0; i < k; i++) {
        clusterSize[i] = 0;
    }
    for (int p = 0; p < data.sizeList; p++) {
        float minDistance = INFINITY;
        int nearestCentroid;

        for (int c = 0; c < k; c++) {
            float distance = getDistance(data.pointList[p], centroids[c]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCentroid = c;
            }
        }
        clusterization[p] = nearestCentroid;
        clusterSize[nearestCentroid]++;
    }
}

Point* initialCentroids(int k, const SetOfPoints& data) {
    if (data.sizeList == k) {
        return data.pointList;
    }
    Point* centroids = (Point*) calloc(k, sizeof(Point));

    int firstIndex = 0;
    copyPoint(&(centroids[0]), &(data.pointList[firstIndex]));

    for (int i = 1; i < k; i++) {
        float maxMinDistance = 0;
        int newCentroidIndex = 0;

        for (int p = 0; p < data.sizeList; p++) {
            float minDistance = INFINITY;
            for (int j = 0; j < i; j++) {
                float distance = getDistance(data.pointList[p], centroids[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            if (minDistance > maxMinDistance) {
                maxMinDistance = minDistance;
                newCentroidIndex = p;
            }
        }
        copyPoint(&(centroids[i]), &(data.pointList[newCentroidIndex]));
    }

    return centroids;
}
