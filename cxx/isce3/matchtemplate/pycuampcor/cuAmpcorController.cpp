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
#include <algorithm>
#include <unistd.h>
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


namespace {

// One controller of a run: its parameters and the images it fills
struct Layer {
    cuAmpcorParameter *param;
    std::unique_ptr<cuArrays<float2>> offsetRun;
    std::unique_ptr<cuArrays<float>> snrRun;
    std::unique_ptr<cuArrays<float3>> covRun;
    std::unique_ptr<cuArrays<float>> corrRun;

    explicit Layer(cuAmpcorParameter *p) : param(p)
    {
        // nWindowsDownRun is defined as numberChunk * numberWindowInChunk
        // It may be bigger than the actual number of windows
        const int down = param->numberChunkDown * param->numberWindowDownInChunk;
        const int across = param->numberChunkAcross * param->numberWindowAcrossInChunk;
        offsetRun.reset(new cuArrays<float2>(down, across));
        snrRun.reset(new cuArrays<float>(down, across));
        covRun.reset(new cuArrays<float3>(down, across));
        corrRun.reset(new cuArrays<float>(down, across));
        offsetRun->allocate();
        snrRun->allocate();
        covRun->allocate();
        corrRun->allocate();
    }

    // first image row of chunk k (reference or secondary)
    int firstRow(int k) const
    {
        return std::min(param->referenceChunkStartPixelDown[k],
                        param->secondaryChunkStartPixelDown[k]);
    }

    // extract the run images and write the output files
    void write() const;
};

void Layer::write() const
{
    cuArrays<float2> offsetImage(param->numberWindowDown, param->numberWindowAcross);
    cuArrays<float> snrImage(param->numberWindowDown, param->numberWindowAcross);
    cuArrays<float3> covImage(param->numberWindowDown, param->numberWindowAcross);
    cuArrays<float> corrImage(param->numberWindowDown, param->numberWindowAcross);
    offsetImage.allocate();
    snrImage.allocate();
    covImage.allocate();
    corrImage.allocate();

    // extraction of the run images to output images
    cuArraysCopyExtract(offsetRun.get(), &offsetImage, make_int2(0,0));
    cuArraysCopyExtract(snrRun.get(), &snrImage, make_int2(0,0));
    cuArraysCopyExtract(covRun.get(), &covImage, make_int2(0,0));
    cuArraysCopyExtract(corrRun.get(), &corrImage, make_int2(0,0));

    /* save the offsets and gross offsets */
    // copy the offset to host
    offsetImage.allocateHost();
    offsetImage.copyToHost();
    // construct the gross offset
    cuArrays<float2> grossOffsetImage(param->numberWindowDown, param->numberWindowAcross);
    grossOffsetImage.allocateHost();
    for(int i=0; i< param->numberWindows; i++)
        grossOffsetImage.hostData[i] = make_float2(param->grossOffsetDown[i], param->grossOffsetAcross[i]);

    // check whether to merge gross offset
    if (param->mergeGrossOffset)
    {
        // if merge, add the gross offsets to offset
        for(int i=0; i< param->numberWindows; i++)
            offsetImage.hostData[i] += grossOffsetImage.hostData[i];
    }
    // output both offset and gross offset
    offsetImage.outputHostToFile(param->offsetImageName);
    grossOffsetImage.outputHostToFile(param->grossOffsetImageName);

    // save the snr/cov images
    snrImage.outputToFile(param->snrImageName);
    covImage.outputToFile(param->covImageName);

    // save the cross-correlation peak
    corrImage.outputToFile(param->corrImageName);
}

// A chunk of a layer
struct Item { int layer, chunk; };

/**
 * Process chunks on the CPU with nThreads threads, one chunk processor per
 * thread and layer; chunks come from nextItem (layer negative when none is
 * left). Returns the number of chunks processed.
 */
int runItemsCPU(std::vector<Layer> &layers, GDALImage *referenceImage,
    GDALImage *secondaryImage, int nThreads,
    const std::function<Item()> &nextItem, const std::function<void()> &chunkDone)
{
    if(nThreads < 1) return 0;
    // Processors are constructed serially since FFTW planning is not
    // thread-safe (fftwf_execute on distinct plans is). nStreams is a CUDA
    // setting and is not used by this CPU port.
    std::vector<std::vector<std::unique_ptr<cuAmpcorChunk>>> chunk(nThreads);
    for(int ist=0; ist<nThreads; ist++)
        for(auto &layer : layers)
            chunk[ist].emplace_back(new cuAmpcorChunk(layer.param, referenceImage,
                secondaryImage, layer.offsetRun.get(), layer.snrRun.get(),
                layer.covRun.get(), layer.corrRun.get()));

    // Chunks write disjoint regions of the *Run images and only read the
    // input images, so they can be processed concurrently.
    std::atomic<int> processed{0};
    std::exception_ptr error = nullptr;
    #pragma omp parallel num_threads(nThreads)
    {
        auto &processors = chunk[omp_get_thread_num()];
        for(Item item = nextItem(); item.layer >= 0; item = nextItem()) {
            const cuAmpcorParameter *param = layers[item.layer].param;
            // exceptions must not leave the OpenMP region: keep the first
            // one, finish the loop, and rethrow after the region
            try {
                // k is the row-major chunk index (down, across)
                processors[item.layer]->run(item.chunk / param->numberChunkAcross,
                                            item.chunk % param->numberChunkAcross);
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

// physical memory in bytes
size_t physicalMemory()
{
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? static_cast<size_t>(pages) * page : 0;
}

} // namespace

/**
 *  Run ampcor
 *
 *
 */
void cuAmpcorController::runAmpcor()
{
    runAmpcorLayers({this});
}

void cuAmpcorController::runAmpcorLayers(const std::vector<cuAmpcorController*>& controllers)
{
    if(controllers.empty()) return;
    // controllers on other images than the first one run in turn
    std::vector<cuAmpcorController*> shared, others;
    for(auto *c : controllers) {
        const auto &p0 = *controllers.front()->param;
        (c->param->referenceImageName == p0.referenceImageName &&
         c->param->secondaryImageName == p0.secondaryImageName
         ? shared : others).push_back(c);
    }
    for(auto *c : others)
        runAmpcorLayers({c});

    cuAmpcorParameter *param0 = shared.front()->param.get();
    // initialize the gdal driver
    GDALAllRegister();
    // reference and secondary images; use band=1 as default
    // TODO: selecting band
    std::cout << "Opening reference image " << param0->referenceImageName << "...\n";
    std::unique_ptr<GDALImage> referenceImage(new GDALImage(param0->referenceImageName, 1, param0->mmapSizeInGB));
    std::cout << "Opening secondary image " << param0->secondaryImageName << "...\n";
    std::unique_ptr<GDALImage> secondaryImage(new GDALImage(param0->secondaryImageName, 1, param0->mmapSizeInGB));
    // read through row caches sharing a fraction of the physical memory
    const size_t cacheBytes = static_cast<size_t>(
        param0->rowCacheMemoryFraction * physicalMemory() / 2);
    referenceImage->enableRowCache(cacheBytes);
    secondaryImage->enableRowCache(cacheBytes);

    std::vector<Layer> layers;
    layers.reserve(shared.size());
    for(auto *c : shared)
        layers.emplace_back(c->param.get());

    // chunks of all layers in the order of their first image row, handed out
    // one at a time to the CPU threads (and the Metal GPU), so that faster
    // processors take more of them
    std::vector<Item> items;
    for(int l = 0; l < static_cast<int>(layers.size()); l++)
        for(int k = 0; k < layers[l].param->numberChunks; k++)
            items.push_back({l, k});
    std::stable_sort(items.begin(), items.end(), [&](const Item &a, const Item &b) {
        return layers[a.layer].firstRow(a.chunk) < layers[b.layer].firstRow(b.chunk);
    });
    const int nItems = static_cast<int>(items.size());
    std::atomic<int> next{0}, nDone{0};
    // thread-safe: shared by the CPU threads and the GPU feeding threads;
    // next may run past nItems, which only yields layer -1
    auto nextItem = [&]() { const int i = next++; return i < nItems ? items[i] : Item{-1, -1}; };
    const int messageInterval = std::max(nItems/10, 1);
    std::mutex messageMutex;
    auto chunkDone = [&]() {
        const int done = ++nDone;
        if(done % messageInterval == 0) {
            std::lock_guard<std::mutex> lock(messageMutex);
            std::cout << "Processed " << done << " out of " << nItems << " chunks" << std::endl;
        }
    };

    int nThreads = isce3::fft::detail::getMaxThreads();
    bool gpu = false;
#ifdef ISCE3_METAL
    gpu = param0->useMetal;
    for(auto &layer : layers)
        gpu = gpu && metalSupported(layer.param);
#endif
    // threads feeding the GPU, each loading its own chunks from the images;
    // two balance chunk loading and CPU processing on Apple M5 (10 cores)
    const int nGpuThreads = gpu ? std::max(1, std::min(nThreads - 1, 2)) : 0;
    nThreads -= nGpuThreads;

    for(auto &layer : layers)
        std::cout << "Total number of windows (azimuth x range):  "
            << layer.param->numberWindowDown << " x " << layer.param->numberWindowAcross
            << " in " << layer.param->numberChunkDown << " x "
            << layer.param->numberChunkAcross << " chunks (window "
            << layer.param->windowSizeHeightRaw << " x "
            << layer.param->windowSizeWidthRaw << ")" << std::endl;
    std::cout << "processed in one pass using " << nThreads << " CPU threads";
    if(gpu) std::cout << " and the Metal GPU (" << nGpuThreads << " feeding threads)";
    std::cout << std::endl;

    std::atomic<int> gpuChunks{0};
    std::exception_ptr gpuError = nullptr;
    std::mutex gpuErrorMutex;
    std::vector<std::thread> gpuThreads;
#ifdef ISCE3_METAL
    std::vector<MetalLayer> metalLayers;
    for(auto &layer : layers)
        metalLayers.push_back({layer.param, layer.offsetRun.get(), layer.snrRun.get(),
                               layer.covRun.get(), layer.corrRun.get()});
    auto nextMetal = [&]() { const Item i = nextItem(); return std::make_pair(i.layer, i.chunk); };
    for(int t = 0; t < nGpuThreads; t++) {
        // each feeding thread runs its own Metal pipelines and pulls chunks
        // from the same queue as the CPU threads
        gpuThreads.emplace_back([&]() {
            try {
                gpuChunks += runAmpcorMetal(metalLayers, referenceImage.get(),
                    secondaryImage.get(), nextMetal, chunkDone);
            }
            catch(...) {
                std::lock_guard<std::mutex> lock(gpuErrorMutex);
                if(!gpuError) gpuError = std::current_exception();
            }
        });
    }
#endif
    std::exception_ptr cpuError = nullptr;
    int cpuChunks = 0;
    try {
        cpuChunks = runItemsCPU(layers, referenceImage.get(), secondaryImage.get(),
                                nThreads, nextItem, chunkDone);
    }
    catch(...) {
        cpuError = std::current_exception();
    }
    for(auto &t : gpuThreads) t.join();
    if(cpuError) std::rethrow_exception(cpuError);
    if(gpuError) std::rethrow_exception(gpuError);
    if(gpu)
        std::cout << "Chunks processed: " << cpuChunks << " on the CPU, "
            << gpuChunks << " on the Metal GPU" << std::endl;

    for(auto &layer : layers)
        layer.write();
}

} // namespace
