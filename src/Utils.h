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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define PPMREADBUFLEN 256

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

static const char *skipSpaces(const char *line) {
	while (*line == ' ' || *line == '\t') {
		line++;
		if (*line == '\0') {
			break;
		}
	}
	return line;
}

static char nextNonSpaceChar(const char *line0) {
	const char *line = skipSpaces(line0);
	return *line;
}

static bool isComment(const char *line) {
	char nextChar = nextNonSpaceChar(line);
	if (nextChar == '\0') {
		return true;
	} else {
		return nextChar == '#';
	}
}

static void parseDimensions(const char *line0, int *width, int *height) {
	const char *line = skipSpaces(line0);
	sscanf(line, "%d %d", width, height);
}

static void parseDimensions(const char *line0, int *width, int *height,
		int *channels) {
	const char *line = skipSpaces(line0);
	sscanf(line, "%d %d %d", width, height, channels);
}

static void parseDepth(const char *line0, int *depth) {
	const char *line = skipSpaces(line0);
	sscanf(line, "%d", depth);
}

static char *File_readLine(FILE* file) {
	static char buffer[PPMREADBUFLEN];
	if (file == NULL) {
		return NULL;
	}
	memset(buffer, 0, PPMREADBUFLEN);

	if (fgets(buffer, PPMREADBUFLEN - 1, file)) {
		return buffer;
	} else {
		return NULL;
	}
}

static char *nextLine(FILE* file) {
	char *line = NULL;
	while ((line = File_readLine(file)) != NULL) {
		if (!isComment(line)) {
			break;
		}
	}
	return line;
}

char* File_read(FILE* file, size_t size, size_t count);

bool File_write(FILE* file, const void *buffer, size_t size, size_t count);

Image* load(const char *filename);

SetOfPoints pixelize(const Image* img);

bool saveRGBimage2D(int k, SetOfPoints* clusters, const char *filename, const Image* img) noexcept(false);

#endif /* UTILS_H_ */
