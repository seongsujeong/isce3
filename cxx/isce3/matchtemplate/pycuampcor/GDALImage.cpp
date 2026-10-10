/**
 * @file  GDALImage.h
 * @brief Implementations of GDALImage class
 *
 */

// my declaration
#include "GDALImage.h"

// dependencies
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace isce3::matchtemplate::pycuampcor {

inline void memcpy2d(void* dst, size_t dst_pitch,
                     void* src, size_t src_pitch, size_t width, int height)
{
    for (int i = 0; i < height; i++) {
        memcpy(dst, src, width);

        dst = (char*) dst + dst_pitch;
        src = (char*) src + src_pitch;
    }
}

/**
 * Cache of row blocks of an image (GDALImage::enableRowCache). Blocks of
 * blockRows rows are read whole by GDAL (one reader at a time); a prefetch
 * thread keeps the blocks after the highest requested one loaded. When the
 * cache is full, the unused block furthest behind the requests (else the
 * one furthest ahead) is dropped; blocks in use are never dropped.
 */
struct GDALImage::RowCache {
    struct Block {
        std::vector<char> data;
        bool ready = false;
    };

    GDALImage &image;
    size_t rowBytes, blockRows, nBlocks, capacity, ahead;
    std::mutex mutex, readMutex;
    std::condition_variable cv;
    std::map<size_t, std::shared_ptr<Block>> blocks;
    size_t wanted = 0;    // highest requested block
    bool stop = false;
    std::thread prefetcher;

    RowCache(GDALImage &img, size_t maxBytes) : image(img)
    {
        rowBytes = static_cast<size_t>(image._width) * image._pixelSize;
        // blocks of about 64 MB, a multiple of the storage block (chunk)
        // height of the source so that every chunk is decoded once
        int blockX = 1, blockY = 1;
        image._poBand->GetBlockSize(&blockX, &blockY);
        const size_t chunkRows = std::max(1, blockY);
        blockRows = std::max<size_t>(16, (64u << 20) / rowBytes);
        blockRows = std::max<size_t>(1, blockRows / chunkRows) * chunkRows;
        nBlocks = (image._height + blockRows - 1) / blockRows;
        capacity = std::max<size_t>(4, maxBytes / (blockRows * rowBytes));
        ahead = std::max<size_t>(1, capacity / 4);
        prefetcher = std::thread([this] { prefetch(); });
    }

    ~RowCache()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        prefetcher.join();
    }

    // drop blocks beyond the capacity (lock held), but not block keep
    void evict(size_t keep)
    {
        while (blocks.size() >= capacity) {
            auto victim = blocks.end();
            for (auto it = blocks.begin(); it != blocks.end(); ++it) {
                if (it->first == keep || !it->second->ready || it->second.use_count() > 1)
                    continue;
                // behind the requests: the furthest behind (first found)
                if (it->first < wanted) { victim = it; break; }
                victim = it;  // ahead: keep the last (furthest ahead)
            }
            if (victim == blocks.end())
                return;  // all in use: grow beyond the capacity
            blocks.erase(victim);
        }
    }

    // load block b (lock held on entry and exit, released while reading)
    std::shared_ptr<Block> load(size_t b, std::unique_lock<std::mutex> &lock)
    {
        evict(b);
        auto block = std::make_shared<Block>();
        blocks[b] = block;
        lock.unlock();
        const size_t row0 = b * blockRows;
        const size_t rows = std::min(blockRows, image._height - row0);
        block->data.resize(rows * rowBytes);
        CPLErr err;
        {
            std::lock_guard<std::mutex> io(readMutex);
            err = image._poBand->RasterIO(GF_Read, 0, static_cast<int>(row0),
                image._width, static_cast<int>(rows), block->data.data(),
                image._width, static_cast<int>(rows), image._dataType, 0, 0);
        }
        lock.lock();
        if (err != CE_None) {
            blocks.erase(b);
            cv.notify_all();
            throw std::runtime_error("GDALImage: reading rows failed");
        }
        block->ready = true;
        cv.notify_all();
        return block;
    }

    // block b, loaded if needed
    std::shared_ptr<Block> get(size_t b)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (b > wanted) {
            wanted = b;
            cv.notify_all();
        }
        auto it = blocks.find(b);
        if (it == blocks.end())
            return load(b, lock);
        auto block = it->second;
        cv.wait(lock, [&] { return block->ready || !blocks.count(b); });
        if (!block->ready)  // its read failed
            throw std::runtime_error("GDALImage: reading rows failed");
        return block;
    }

    void prefetch()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stop) {
            // first missing block ahead of the requests, if any
            size_t b = wanted + 1;
            const size_t last = std::min(nBlocks, wanted + 1 + ahead);
            while (b < last && blocks.count(b))
                b++;
            if (b < last) {
                try {
                    load(b, lock);
                } catch (...) {
                    // the request of the block reports the error
                }
                continue;
            }
            cv.wait(lock);
        }
    }

    void copyRows(char *dst, size_t row0, size_t col0, size_t rows, size_t cols)
    {
        const size_t pixel = image._pixelSize;
        for (size_t r = row0; r < row0 + rows;) {
            const size_t b = r / blockRows;
            const auto block = get(b);
            const size_t end = std::min(row0 + rows, (b + 1) * blockRows);
            for (; r < end; r++, dst += cols * pixel)
                std::memcpy(dst, block->data.data() + (r - b * blockRows) * rowBytes +
                            col0 * pixel, cols * pixel);
        }
    }
};

void GDALImage::enableRowCache(size_t maxBytes)
{
    _rowCache.reset();
    if (maxBytes > 0)
        _rowCache.reset(new RowCache(*this, maxBytes));
}

/**
 * Constructor
 * @brief Create a GDAL image object
 * @param filename a std::string with the raster image file name
 * @param band the band number
 * @param cacheSizeInGB read buffer size in GigaBytes
 * @param useMmap whether to use memory map
 */
GDALImage::GDALImage(std::string filename, int band, int cacheSizeInGB, int useMmap)
   : _useMmap(useMmap)
{
    // open the file as dataset
    _poDataset = (GDALDataset *) GDALOpen(filename.c_str(), GA_ReadOnly);
    // if something is wrong, throw an exception
    // GDAL reports the error message
    if(!_poDataset)
        throw;

    // check the band info
    int count = _poDataset->GetRasterCount();
    if(band > count)
    {
        std::cout << "The desired band " << band << " is greater than " << count << " bands available";
        throw;
    }

    // get the desired band
    _poBand = _poDataset->GetRasterBand(band);
    if(!_poBand)
        throw;

     // get the width(x), and height(y)
    _width = _poBand->GetXSize();
    _height = _poBand->GetYSize();

    _dataType = _poBand->GetRasterDataType();
    // determine the image type
    _isComplex = GDALDataTypeIsComplex(_dataType);
    // determine the pixel size in bytes
    _pixelSize = GDALGetDataTypeSizeBytes(_dataType);

    _bufferSize = 1024*1024*cacheSizeInGB;

    // checking whether using memory map
    if(_useMmap) {

       char **papszOptions = NULL;
        // if cacheSizeInGB = 0, use default
        // else set the option
        if(cacheSizeInGB > 0)
            papszOptions = CSLSetNameValue( papszOptions,
                "CACHE_SIZE",
                std::to_string(_bufferSize).c_str());

        // space between two lines
        GIntBig pnLineSpace;
        // set up the virtual mem buffer
        _poBandVirtualMem =  GDALGetVirtualMemAuto(
            static_cast<GDALRasterBandH>(_poBand),
            GF_Read,
            &_pixelSize,
            &pnLineSpace,
            papszOptions);
        CSLDestroy(papszOptions);
        // formats that cannot be memory mapped (e.g. chunked, compressed
        // HDF5) are read with GDAL instead, best through a row cache
        if(_poBandVirtualMem)
            _memPtr = CPLVirtualMemGetAddr(_poBandVirtualMem);
        else {
            CPLErrorReset();
            _useMmap = 0;
            _pixelSize = GDALGetDataTypeSizeBytes(_dataType);
        }
    }
    if(!_useMmap) { // use a buffer
        _memPtr = (void*) malloc(_bufferSize);
    }
    // make sure memPtr is not Null
    if (!_memPtr)
    {
        std::cout << "unable to locate the memory buffer\n";
        throw;
    }
    // all done
}


/**
 * Load a tile of data h_tile x w_tile from CPU to GPU
 * @param dArray pointer for array in device memory
 * @param h_offset Down/Height offset
 * @param w_offset Across/Width offset
 * @param h_tile Down/Height tile size
 * @param w_tile Across/Width tile size
 * @note Need to use size_t type to pass the parameters to cudaMemcpy2D correctly
 */
void GDALImage::loadToDevice(void *dArray, size_t h_offset, size_t w_offset,
    size_t h_tile, size_t w_tile)
{

    size_t tileStartOffset = (h_offset*_width + w_offset)*_pixelSize;

    char * startPtr = (char *)_memPtr ;
    startPtr += tileStartOffset;

    if (_rowCache) {
        _rowCache->copyRows(static_cast<char*>(dArray), h_offset, w_offset, h_tile, w_tile);
    }
    else if (_useMmap) {
        // direct copy from memory map buffer to device memory
        memcpy2d(dArray,      // dst
            w_tile*_pixelSize,                    // dst pitch
            startPtr,                             // src
            _width*_pixelSize,                    // src pitch
            w_tile*_pixelSize,                    // width in Bytes
            h_tile);                              // height
    }
    else { // use a cpu buffer to load image data to gpu

        // get the total tile size in bytes
        size_t tileSize = h_tile*w_tile*_pixelSize;
        // if the size is bigger than existing buffer, reallocate
        if (tileSize > _bufferSize) {
            // TODO: fit the pagesize
            _bufferSize = tileSize;
            free(_memPtr);
            _memPtr = (void*) malloc(_bufferSize);
        }
        // copy from file to buffer
        CPLErr err = _poBand->RasterIO(GF_Read, //eRWFlag
            w_offset, h_offset,  //nXOff, nYOff
            w_tile, h_tile,  // nXSize, nYSize
            _memPtr, // pData
            w_tile, h_tile, // nBufXSize, nBufYSize
            _dataType, //eBufType
            0, 0 //nPixelSpace, nLineSpace in pData
            );
        if(err != CE_None)
            throw; // throw if reading error occurs; message reported by GDAL

        // copy from buffer
        memcpy(dArray, _memPtr, tileSize);
    }
    // all done
}

/// destructor
GDALImage::~GDALImage()
{
    // stop the row cache prefetching before closing the dataset
    _rowCache.reset();
    // free the virtual memory or the buffer
    if(_poBandVirtualMem)
        CPLVirtualMemFree(_poBandVirtualMem);
    else
        free(_memPtr);
    // free the GDAL Dataset, close the file
    delete _poDataset;
}

} // namespace
