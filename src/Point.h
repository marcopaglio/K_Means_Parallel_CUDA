/*
 * Point.h
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#ifndef POINT_H_
#define POINT_H_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

using namespace std;

typedef struct {
    float* coordinates;
    int dimension;
    string metadata;
} Point;

float getCoordinateAt(const Point& p, int pos) noexcept(false);

void copyPoint(Point* copy, const Point* original);

void setDimension(Point* p, int dim);

void setCoordinates(Point* p, float* coordinates);

void setMetadata(Point* p, const string& meta);

__host__ float getDistance(const Point& p1, const Point& p2) noexcept(false); //TODO anche __device__

void setAttributes(Point* p, float* c, int d, const string& s);

#endif /* POINT_H_ */
