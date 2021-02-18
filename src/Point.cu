/*
 * Point.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "Point.h"
#include <cmath>

using namespace std;

__host__ __device__ float getCoordinateAt(const Point& p, int pos) noexcept(false) {
    /*if (pos < 0 || pos >= p.dimension) {
        throw out_of_range("Cannot enter this position");
    }*/
    return p.coordinates[pos];
}

// copyPoint fa la copia profonda
void copyPoint(Point* copy, const Point* original) {
    setDimension(copy, (original)->dimension);
    float* copyCoordinates = (float *) calloc((original)->dimension, sizeof (float));
    for (int i = 0; i < (original)->dimension; i++) {
        copyCoordinates[i] = (original)->coordinates[i];
    }
    setCoordinates(copy, copyCoordinates);
    setMetadata(copy, (original)->metadata);
}

void Point_init(Point* copy, float* original, unsigned int dimOriginal, string meta) {
    setDimension(copy, dimOriginal);
    float* copyCoordinates = (float *) calloc(dimOriginal, sizeof (float));
    for (int i = 0; i < dimOriginal; i++) {
        copyCoordinates[i] = original[i];
    }
    setCoordinates(copy, copyCoordinates);
    setMetadata(copy, meta);
}


void setDimension(Point* p, int dim) {
    (p)->dimension = dim;
}

void setCoordinates(Point* p, float* coordinates) {
    (p)->coordinates = coordinates;
}

void setMetadata(Point* p, const string& meta) {
    if ((p)->metadata != meta) {
        (p)->metadata = meta;
    }     // else lancia fault segmentation SIGSEGV error

}

__host__ __device__ float getDistance(const Point& p1, const Point& p2) /*noexcept(false)*/ {
    /*if (p1.dimension != p2.dimension) {
        throw invalid_argument("Distance is calculated only for same dimension points");
    }*/
    float sum = 0;
    float difference;
    for (unsigned int i = 0; i < p1.dimension; i++) {
        difference = p1.coordinates[i] - p2.coordinates[i];
        sum += difference * difference;
    }
    return sum;
}

__host__ __device__ float getDistanceByCoordinates(const float* p1, const float* p2, unsigned int dim) {
    float sum = 0;
    float difference;
    for (unsigned int i = 0; i < dim; i++) {
        difference = p1[i] - p2[i];
        sum += difference * difference;
    }
    return sum;
}

void setAttributes(Point* p, float* c, int d, const string& s) {
    setCoordinates(p, c);
    setDimension(p, d);
    setMetadata(p, s);
}

void Point_delete(Point* p) {
	if (p != NULL) {
		if (p->coordinates != NULL) {
			free(p->coordinates);
		}
	}
}
