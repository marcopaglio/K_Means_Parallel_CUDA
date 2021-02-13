/*
 * Image.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "Image.h"
#include "Utils.h"
#include <iostream>
#include <cassert>

Image* Image_new(int width, int height, int channels, unsigned char *data) {
	Image* img;

	img = (Image*) malloc(sizeof(Image));

	Image_setWidth(img, width);
	Image_setHeight(img, height);
	Image_setChannels(img, channels);
	Image_setPitch(img, width * channels);

	Image_setData(img, data);
	return img;
}

Image* Image_new(int width, int height, int channels) {
	unsigned char *data = (unsigned char*) malloc(sizeof(unsigned char) * width * height * channels);
	return Image_new(width, height, channels, data);
}

Image* Image_new(int width, int height) {
	return Image_new(width, height, Image_channels);
}

void Image_delete(Image* img) {
	if (img != NULL) {
		if (Image_getData(img) != NULL) {
			free(Image_getData(img));
		}
		free(img);
	}
}

void Image_setPixel(Image* img, int x, int y, int c, unsigned char val) {
	unsigned char *data = Image_getData(img);
	int channels = Image_getChannels(img);
	int pitch = Image_getPitch(img);

	data[y * pitch + x * channels + c] = val;

	return;
}

unsigned char Image_getPixel(Image* img, int x, int y, int c) {
	unsigned char *data = Image_getData(img);
	int channels = Image_getChannels(img);
	int pitch = Image_getPitch(img);

	return data[y * pitch + x * channels + c];
}
