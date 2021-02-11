/*
 * Utils.cpp
 *
 *  Created on: 8 feb 2021
 *      Author: marco
 */

#include "Utils.h"

using namespace std;


char* File_read(FILE* file, size_t size, size_t count) {
	size_t res;
	char *buffer;
	size_t bufferLen;

	if (file == NULL) {
		return NULL;
	}

	bufferLen = size * count + 1;
	buffer = (char*) malloc(sizeof(char) * bufferLen);

	res = fread(buffer, size, count, file);
	// make valid C string
	buffer[size * res] = '\0';

	return buffer;
}

bool File_write(FILE* file, const void *buffer, size_t size, size_t count) {
	if (file == NULL) {
		return false;
	}

	size_t res = fwrite(buffer, size, count, file);
	if (res != count) {
		printf("ERROR: Failed to write data to PPM file");
	}

	return true;
}

Image* load(const char *filename){
	Image* img;
	FILE* file;
	char *header;
	char *line;
	int ii, jj, kk, channels;
	int width, height, depth;
	unsigned char *charData, *charIter;
	float *imgData, *floatIter;
	float scale;

	img = NULL;

	file = fopen(filename, "rb");
	if (file == NULL) {
		printf("Could not open %s\n", filename);
		goto cleanup;
	}

	header = File_readLine(file);
	if (header == NULL) {
		printf("Could not read from %s\n", filename);
		goto cleanup;
	} else if (strcmp(header, "P6") != 0 && strcmp(header, "P6\n") != 0
			&& strcmp(header, "P5") != 0 && strcmp(header, "P5\n") != 0
			&& strcmp(header, "S6") != 0 && strcmp(header, "S6\n") != 0) {
		printf("Could not find magic number for %s\n", filename);
		goto cleanup;
	}

	// P5 are monochrome while P6/S6 are RGB
	// S6 needs to parse number of channels out of file
	if (strcmp(header, "P5") == 0 || strcmp(header, "P5\n") == 0) {
		channels = 1;
		line = nextLine(file);
		parseDimensions(line, &width, &height);
	} else if (strcmp(header, "P6") == 0 || strcmp(header, "P6\n") == 0) {
		channels = 3;
		line = nextLine(file);
		parseDimensions(line, &width, &height);
	} else {
		line = nextLine(file);
		parseDimensions(line, &width, &height, &channels);
	}

	// the line now contains the depth information
	line = nextLine(file);
	parseDepth(line, &depth);

	// the rest of the lines contain the data in binary format
	charData = (unsigned char *) File_read(file,
			width * channels * sizeof(unsigned char), height);

	img = Image_new(width, height, channels);

	imgData = Image_getData(img);

	charIter = charData;
	floatIter = imgData;

	scale = 1.0f / ((float) depth);

	for (ii = 0; ii < height; ii++) {
		for (jj = 0; jj < width; jj++) {
			for (kk = 0; kk < channels; kk++) {
				*floatIter = ((float) *charIter) * scale;
				floatIter++;
				charIter++;
			}
		}
	}

	cleanup: fclose(file);
	return img;
}

SetOfPoints pixelize(const Image* img) {
	int height = Image_getHeight(img);
	int width = Image_getWidth(img);
    int channels = Image_getChannels(img);

    SetOfPoints data;
    data.pointList = (Point *) calloc(width * height, sizeof (Point));
    data.sizeList = width * height;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            auto coordinates = (float *) calloc(channels, sizeof(float));
            for (int z = 0; z < channels; z++) {
            	coordinates[z] = Image_getData(img)[y * width * channels + x * channels + z] * 255;
            }
            insertPoint(&data,
                        Point{.coordinates = coordinates, .dimension = channels, .metadata = to_string(x) + "," + to_string(y)},
                        y * width + x);
        }
    }
    return data;
}

bool saveRGBimage2D(int k, SetOfPoints* clusters, const char *filename, const Image* img) noexcept(false) {
    if (clusters == nullptr) {
        throw invalid_argument("Clusters can't be null");
    }
	int width = Image_getWidth(img);
	int height = Image_getHeight(img);
	int channels = Image_getChannels(img);
    Image* outputImage = Image_new(width, height, channels);

	int depth = 255;

    float color[channels];
    for (int i = 0; i < k; i++) {
        SetOfPoints cluster = clusters[i];
        Point center = getCenter(cluster);
        for (int d = 0; d < channels; d++) {
        	color[d] = getCoordinateAt(center, d) / 255;
        }

        free(center.coordinates);

        for (int j = 0; j < cluster.sizeList; j++) {
            string position = cluster.pointList[j].metadata;
            int splitIndex = position.find_first_of(',', 0); //WORKS ONLY FOR 2D IMAGES
            int x = stoi(position.substr(0, splitIndex));
            int y = stoi(position.substr(splitIndex + 1, position.length() - 1));
            for (int z = 0; z < channels; z++) {
				Image_getData(outputImage)[y * width * channels + x * channels + z] = color[z];
            }
        }
    }

	int ii;
	int jj;
	int kk;

	FILE* file;
	float *floatIter;
	unsigned char *charData;
	unsigned char *charIter;

	file = fopen(filename, "wb+");
	if (file == NULL) {
		printf("Could not open %s in mode %s\n", filename, "wb+");
		return false;
	}

	if (channels == 1) {
		fprintf(file, "P5\n");
	} else {
		fprintf(file, "P6\n");
	}
	fprintf(file, "#Created via PPM Export\n");
	fprintf(file, "%d %d\n", width, height);
	fprintf(file, "%d\n", depth);

	charData = (unsigned char*) malloc(
			sizeof(unsigned char) * width * height * channels);

	charIter = charData;
	floatIter = Image_getData(outputImage);

	for (ii = 0; ii < height; ii++) {
		for (jj = 0; jj < width; jj++) {
			for (kk = 0; kk < channels; kk++) {
				*charIter = (unsigned char) ceil(
						clamp(*floatIter, 0, 1) * depth);
				floatIter++;
				charIter++;
			}
		}
	}

	bool writeResult = File_write(file, charData,
			width * channels * sizeof(unsigned char), height);

	free(outputImage);
	free(charData);
	fflush(file);
	fclose(file);

	return true;
}



