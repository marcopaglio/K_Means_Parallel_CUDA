/*
 * SetOfPoints.h
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#ifndef SETOFPOINTS_H_
#define SETOFPOINTS_H_

#include "Point.h"

typedef struct {
    Point* pointList;
    int sizeList;
} SetOfPoints;

__host__ SetOfPoints* SetOfPoints_new(Point* pl, int s);

__host__ void setSizeList(SetOfPoints* cluster, int s) noexcept(false);

__host__ void setPointList(SetOfPoints* cluster, Point* pl);

__host__ void setAttributes(SetOfPoints* cluster, Point* pl, int s);

void insertPoint(SetOfPoints* cluster, const Point& p, int pos);

__host__ Point getCenter(const SetOfPoints& cluster) noexcept(false);

float getDiameter(const SetOfPoints& cluster);

#endif /* SETOFPOINTS_H_ */
