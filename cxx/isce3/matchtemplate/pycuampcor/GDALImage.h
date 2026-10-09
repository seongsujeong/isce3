/**
 * @file GDALImage.h
 * @brief Interface with GDAL vrt driver
 *
 * To read image file with the GDAL vrt driver, including SLC, GeoTIFF images
 * @warning Only single precision images are supported: complex(pixelOffset=8) or real(pixelOffset=4).
 * @warning Only single band file is currently supported.
 */

// code guard
#ifndef __GDALIMAGE_H
#define __GDALIMAGE_H

// dependencies
#include <memory>
#include <string>
#include <gdal_priv.h>
#include <cpl_conv.h>

namespace isce3::matchtemplate::pycuampcor {

class GDALImage{
public:
    // specify the types
    using size_t = std::size_t;

private:
    int _height;      ///< image height
    int _width;       ///< image width

    void * _memPtr = NULL; ///< pointer to buffer

    int _pixelSize; ///< pixel size in bytes

    int _isComplex; ///< whether the image is complex

    size_t _bufferSize; ///< buffer size
    int _useMmap;   ///< whether to use memory map

    // GDAL temporary objects
    GDALDataType _dataType;
    CPLVirtualMem * _poBandVirtualMem = NULL;
    GDALDataset * _poDataset = NULL;
    GDALRasterBand * _poBand = NULL;

public:
    //disable default constructor
    GDALImage() = delete;
    // constructor
    GDALImage(std::string fn, int band=1, int cacheSizeInGB=0, int useMmap=1);
    // destructor
    ~GDALImage();

    // get class properties
    void * getmemPtr()
    {
        return(_memPtr);
    }

    int getHeight() {
        return (_height);
    }

    int getWidth()
    {
        return (_width);
    }

    int getPixelSize()
    {
        return _pixelSize;
    }

    bool isComplex()
    {
        return _isComplex;
    }

    // load data from cpu buffer to gpu
    void loadToDevice(void *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile);

    /**
     * Read through an in-memory cache of row blocks instead of the memory
     * map: blocks are read with large sequential reads, the ones ahead of
     * the latest request in a background thread, and the ones furthest
     * behind are dropped beyond maxBytes. Suits passes over the image in
     * row order larger than the memory (memory-map page faults are slow).
     * Thread-safe. maxBytes 0 keeps the memory map.
     */
    void enableRowCache(size_t maxBytes);

private:
    struct RowCache;
    std::unique_ptr<RowCache> _rowCache;

};

} // namespace

#endif //__GDALIMAGE_H
// end of file
