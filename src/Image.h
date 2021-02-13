/*
 * Image.h
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#ifndef IMAGE_H_
#define IMAGE_H_

typedef struct {
	int width;
	int height;
	int channels;
	int pitch;
	unsigned char *data;
} Image;

#define Image_channels 3

#define Image_getWidth(img) ((img)->width)
#define Image_getHeight(img) ((img)->height)
#define Image_getChannels(img) ((img)->channels)
#define Image_getPitch(img) ((img)->pitch)
#define Image_getData(img) ((img)->data)

#define Image_setWidth(img, val) (Image_getWidth(img) = val)
#define Image_setHeight(img, val) (Image_getHeight(img) = val)
#define Image_setChannels(img, val) (Image_getChannels(img) = val)
#define Image_setPitch(img, val) (Image_getPitch(img) = val)
#define Image_setData(img, val) (Image_getData(img) = val)

Image* Image_new(int width, int height, int channels, unsigned char *data);
Image* Image_new(int width, int height, int channels);
Image* Image_new(int width, int height);
unsigned char Image_getPixel(Image* img, int x, int y, int c);
void Image_setPixel(Image* img, int x, int y, int c, unsigned char val);
void Image_delete(Image* img);

#endif /* IMAGE_H_ */
