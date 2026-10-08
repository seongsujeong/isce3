/**
 * @file cuAmpcorController.cu
 * @brief Implementations of cuAmpcorController
 */

// my declaration
#include "cuAmpcorController.h"

// dependencies
#include "GDALImage.h"
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cuAmpcorChunk.h"
#include "cuAmpcorUtil.h"
#include "cuMetal.h"
#include <atomic>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <isce3/fft/detail/Threads.h>
#ifdef _OPENMP
#include <omp.h>
#else
// without OpenMP the parallel region below runs on one thread
static int omp_get_thread_num() { return 0; }
#endif
#include "float2.h"

namespace isce3::matchtemplate::pycuampcor {

// constructor
cuAmpcorController::cuAmpcorController()
{
    // create a new set of parameters
    param.reset(new cuAmpcorParameter());
}


/**
 * Process chunks on the CPU with nThreads threads, one chunk processor per
 * thread; chunk indices come from nextChunk (negative when none is left).
 * Returns the number of chunks processed.
 */
static int runChunksCPU(cuAmpcorParameter *param, GDALImage *referenceImage,
    GDALImage *secondaryImage, cuArrays<float2> *offsetImageRun,
    cuArrays<float> *snrImageRun, cuArrays<float3> *covImageRun,
    cuArrays<float> *corrImageRun, int nThreads,
    const std::function<int()> &nextChunk, const std::function<void()> &chunkDone)
{
    if(nThreads < 1) return 0;
    // Processors are constructed serially since FFTW planning is not
    // thread-safe (fftwf_execute on distinct plans is). nStreams is a CUDA
    // setting and is not used by this CPU port.
    std::vector<std::unique_ptr<cuAmpcorChunk>> chunk(nThreads);
    for(int ist=0; ist<nThreads; ist++)
        chunk[ist].reset(new cuAmpcorChunk(param, referenceImage, secondaryImage,
            offsetImageRun, snrImageRun, covImageRun, corrImageRun));

    // Chunks write disjoint regions of the *Run images and only read the
    // (mmap'ed) input images, so they can be processed concurrently.
    std::atomic<int> processed{0};
    std::exception_ptr error = nullptr;
    #pragma omp parallel num_threads(nThreads)
    {
        cuAmpcorChunk &processor = *chunk[omp_get_thread_num()];
        // k is the row-major chunk index (down, across)
        for(int k = nextChunk(); k >= 0; k = nextChunk()) {
            // exceptions must not leave the OpenMP region: keep the first
            // one, finish the loop, and rethrow after the region
            try {
                processor.run(k / param->numberChunkAcross, k % param->numberChunkAcross);
            }
            catch(...) {
                #pragma omp critical
                if(!error) error = std::current_exception();
            }
            processed++;
            chunkDone();
        }
    }
    if(error) std::rethrow_exception(error);
    return processed;
}

/**
 *  Run ampcor
 *
 *
 */
void cuAmpcorController::runAmpcor()
{
    // initialize the gdal driver
    GDALAllRegister();
    // reference and secondary images; use band=1 as default
    // TODO: selecting band
    std::cout << "Opening reference image " << param->referenceImageName << "...\n";
    GDALImage *referenceImage = new GDALImage(param->referenceImageName, 1, param->mmapSizeInGB);
    std::cout << "Opening secondary image " << param->secondaryImageName << "...\n";
    GDALImage *secondaryImage = new GDALImage(param->secondaryImageName, 1, param->mmapSizeInGB);

    cuArrays<float2> *offsetImage, *offsetImageRun;
    cuArrays<float> *snrImage, *snrImageRun;
    cuArrays<float3> *covImage, *covImageRun;
    cuArrays<float> *corrImage, *corrImageRun;

    // nWindowsDownRun is defined as numberChunk * numberWindowInChunk
    // It may be bigger than the actual number of windows
    int nWindowsDownRun = param->numberChunkDown * param->numberWindowDownInChunk;
    int nWindowsAcrossRun = param->numberChunkAcross * param->numberWindowAcrossInChunk;

    offsetImageRun = new cuArrays<float2>(nWindowsDownRun, nWindowsAcrossRun);
    offsetImageRun->allocate();

    snrImageRun = new cuArrays<float>(nWindowsDownRun, nWindowsAcrossRun);
    snrImageRun->allocate();

    covImageRun = new cuArrays<float3>(nWindowsDownRun, nWindowsAcrossRun);
    covImageRun->allocate();

    corrImageRun = new cuArrays<float>(nWindowsDownRun, nWindowsAcrossRun);
    corrImageRun->allocate();

    // Offset fields.
    offsetImage = new cuArrays<float2>(param->numberWindowDown, param->numberWindowAcross);
    offsetImage->allocate();

    // SNR.
    snrImage = new cuArrays<float>(param->numberWindowDown, param->numberWindowAcross);
    snrImage->allocate();

    // Variance.
    covImage = new cuArrays<float3>(param->numberWindowDown, param->numberWindowAcross);
    covImage->allocate();

    // Cross-correlation peak
    corrImage = new cuArrays<float>(param->numberWindowDown, param->numberWindowAcross);
    corrImage->allocate();

    // chunks are handed out one at a time to the CPU threads (and the Metal
    // GPU), so faster processors take more of them
    const int nChunks = param->numberChunkDown * param->numberChunkAcross;
    std::atomic<int> next{0}, nDone{0};
    // thread-safe: shared by the CPU threads and the GPU feeding threads;
    // next may run past nChunks, which only yields -1
    auto nextChunk = [&]() { const int k = next++; return k < nChunks ? k : -1; };
    const int messageInterval = std::max(nChunks/10, 1);
    std::mutex messageMutex;
    auto chunkDone = [&]() {
        const int done = ++nDone;
        if(done % messageInterval == 0) {
            std::lock_guard<std::mutex> lock(messageMutex);
            std::cout << "Processed " << done << " out of " << nChunks << " chunks" << std::endl;
        }
    };

    int nThreads = isce3::fft::detail::getMaxThreads();
    bool gpu = false;
#ifdef ISCE3_METAL
    gpu = param->useMetal && metalSupported(param.get());
#endif
    // threads feeding the GPU, each loading its own chunks from the images;
    // two balance chunk loading and CPU processing on Apple M5 (10 cores)
    const int nGpuThreads = gpu ? std::max(1, std::min(nThreads - 1, 2)) : 0;
    nThreads -= nGpuThreads;

    std::cout << "Total number of windows (azimuth x range):  "
        << param->numberWindowDown << " x " << param->numberWindowAcross << std::endl;
    std::cout << "to be processed in the number of chunks: "
        << param->numberChunkDown << " x " << param->numberChunkAcross
        << " using " << nThreads << " CPU threads";
    if(gpu) std::cout << " and the Metal GPU (" << nGpuThreads << " feeding threads)";
    std::cout << std::endl;

    std::atomic<int> gpuChunks{0};
    std::exception_ptr gpuError = nullptr;
    std::mutex gpuErrorMutex;
    std::vector<std::thread> gpuThreads;
#ifdef ISCE3_METAL
    for(int t = 0; t < nGpuThreads; t++) {
        // each feeding thread runs its own Metal pipeline and pulls chunks
        // from the same counter as the CPU threads
        gpuThreads.emplace_back([&]() {
            try {
                gpuChunks += runAmpcorMetal(param.get(), referenceImage, secondaryImage,
                    offsetImageRun, snrImageRun, covImageRun, corrImageRun,
                    nextChunk, chunkDone);
            }
            catch(...) {
                std::lock_guard<std::mutex> lock(gpuErrorMutex);
                if(!gpuError) gpuError = std::current_exception();
            }
        });
    }
#endif
    const int cpuChunks = runChunksCPU(param.get(), referenceImage, secondaryImage,
        offsetImageRun, snrImageRun, covImageRun, corrImageRun, nThreads,
        nextChunk, chunkDone);
    for(auto &t : gpuThreads) t.join();
    if(gpuError) std::rethrow_exception(gpuError);
    if(gpu)
        std::cout << "Chunks processed: " << cpuChunks << " on the CPU, "
            << gpuChunks << " on the Metal GPU" << std::endl;

    // extraction of the run images to output images
    cuArraysCopyExtract(offsetImageRun, offsetImage, make_int2(0,0));
    cuArraysCopyExtract(snrImageRun, snrImage, make_int2(0,0));
    cuArraysCopyExtract(covImageRun, covImage, make_int2(0,0));
    cuArraysCopyExtract(corrImageRun, corrImage, make_int2(0,0));

    /* save the offsets and gross offsets */
    // copy the offset to host
    offsetImage->allocateHost();
    offsetImage->copyToHost();
    // construct the gross offset
    cuArrays<float2> *grossOffsetImage = new cuArrays<float2>(param->numberWindowDown, param->numberWindowAcross);
    grossOffsetImage->allocateHost();
    for(int i=0; i< param->numberWindows; i++)
        grossOffsetImage->hostData[i] = make_float2(param->grossOffsetDown[i], param->grossOffsetAcross[i]);

    // check whether to merge gross offset
    if (param->mergeGrossOffset)
    {
        // if merge, add the gross offsets to offset
        for(int i=0; i< param->numberWindows; i++)
            offsetImage->hostData[i] += grossOffsetImage->hostData[i];
    }
    // output both offset and gross offset
    offsetImage->outputHostToFile(param->offsetImageName);
    grossOffsetImage->outputHostToFile(param->grossOffsetImageName);
    delete grossOffsetImage;

    // save the snr/cov images
    snrImage->outputToFile(param->snrImageName);
    covImage->outputToFile(param->covImageName);

    // save the cross-correlation peak
    corrImage->outputToFile(param->corrImageName);

    // Delete arrays.
    delete offsetImage;
    delete snrImage;
    delete covImage;
    delete corrImage;

    delete offsetImageRun;
    delete snrImageRun;
    delete covImageRun;
    delete corrImageRun;

    delete referenceImage;
    delete secondaryImage;

}

} // namespace
