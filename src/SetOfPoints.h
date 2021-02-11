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

SetOfPoints* SetOfPoints_new(Point* pl, int s);

void setSizeList(SetOfPoints* cluster, int s) noexcept(false);

void setPointList(SetOfPoints* cluster, Point* pl);

void setAttributes(SetOfPoints* cluster, Point* pl, int s);

void insertPoint(SetOfPoints* cluster, const Point& p, int pos);

Point getCenter(const SetOfPoints& cluster) noexcept(false);

float getDiameter(const SetOfPoints& cluster);

#endif /* SETOFPOINTS_H_ */
