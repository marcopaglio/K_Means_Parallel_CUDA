/*
 * SetOfPoints.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "SetOfPoints.h"

SetOfPoints* SetOfPoints_new(Point* pl, int s) {
    SetOfPoints* cluster;
    cluster = (SetOfPoints *) calloc(s, sizeof(SetOfPoints));

    setSizeList(cluster, s);
    setPointList(cluster, pl);

    return cluster;
}

void setSizeList(SetOfPoints* cluster, int s) noexcept(false) {
    if (s < 0) {
        throw invalid_argument("Size must be not negative");
    }
    (cluster)->sizeList = s;
}

void setPointList(SetOfPoints* cluster, Point* pl) {
    (cluster)->pointList = pl;
}

void setAttributes(SetOfPoints* cluster, Point* pl, int s) {
    setPointList(cluster, pl);
    setSizeList(cluster, s);
}

void insertPoint(SetOfPoints* cluster, const Point& p, int pos) {
    if (pos < 0 || pos >= (cluster)->sizeList) {
        throw out_of_range("Cannot enter this position");
    }
    ((cluster)->pointList)[pos] = p;
}

Point getCenter(const SetOfPoints& cluster) noexcept(false) {
    if (cluster.sizeList == 0) {
        throw invalid_argument("Cluster has empty list of point");
    }
    int dimension = cluster.pointList[0].dimension;
    float* coordinates = (float *) calloc(dimension, sizeof (float));
    for (int i = 0; i < dimension; i++) {
        coordinates[i] = 0;
        for (int p = 0; p < cluster.sizeList; p++) {
            coordinates[i] += cluster.pointList[p].coordinates[i];
        }
        coordinates[i] /= (float) cluster.sizeList;
    }

    Point center = {.coordinates = coordinates, .dimension = dimension};
    return center;
}

float getDiameter(const SetOfPoints& cluster) {
    float max = 0;
    for (int i = 0; i < cluster.sizeList; i++) {
        Point p1 = cluster.pointList[i];
        for (int j = i+1; j < cluster.sizeList; j++) {
            Point p2 = cluster.pointList[j];
            float distance = getDistance(p1, p2);
            if (distance > max) {
                max = distance;
            }
        }
    }
    return max;
}

