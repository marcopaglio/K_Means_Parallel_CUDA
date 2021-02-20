/*
 * Utils.h
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "SetOfPoints.h"
#include "Point.h"
#include "Image.h"

template<typename T>
static inline T _abs(const T &a) {
	return a < 0 ? -a : a;
}

static inline bool almostEqualFloat(float A, float B, float eps) {
	if (A == 0) {
		return _abs(B) < eps;
	} else if (B == 0) {
		return _abs(A) < eps;
	} else {
#if 0
		float d = max(_abs(A), _abs(B));
		float g = (_abs(A - B) / d);
#else
		float g = _abs(A - B);
#endif
		if (g <= eps) {
			return true;
		} else {
			return false;
		}
	}
}

static inline bool almostEqualFloat(float A, float B) {
	return almostEqualFloat(A, B, 0.2f);
}

static inline bool almostUnequalFloat(float a, float b) {
	return !almostEqualFloat(a, b);
}

static inline float _min(float x, float y) {
	return x < y ? x : y;
}

static inline float _max(float x, float y) {
	return x > y ? x : y;
}

static inline float clamp(float x, float start, float end) {
	return _min(_max(x, start), end);
}

/*
 * Calculates the number of cores of the device
 */
int getSPcores(cudaDeviceProp devProp);

/*
 * Load a JPEG image and return it as Image
 * @param filename path to JPEG
 * @return Image
 */
Image* loadJPEG(const char *filename);

/*
 * Transform an Image in a SetOfPoints (where each point has metadata relative to its position as pixel)
 * @param img reference to the Image to pixelize
 * @return SetOfPoints
 */
SetOfPoints pixelize(const Image* img);

/*
 * Save an array of clusters as PNG image
 * @param clusters reference to array of clusters
 * @param k number of clusters
 * @param filename path to PNG
 * @param width width of image
 * @param height height of image
 * @return TRUE if success, false otherwise
 */
bool savePNG(SetOfPoints* clusters, int k, const char *filename, int width, int height);

#endif /* UTILS_H_ */
