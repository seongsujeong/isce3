/**
 * @file  cuMetal.mm
 * @brief Metal (Apple GPU) implementation of the ampcor chunk pipeline
 *
 * A chunk runs as one command buffer of the kernels in cuAmpcor.metal, in
 * the order of cuAmpcorChunk::run; each kernel mirrors the CPU function it
 * replaces. The CPU only copies the chunk data out of the memory-mapped
 * images and sets the window offsets. Several chunk processors (slots)
 * rotate so that the CPU prepares a chunk while the GPU runs the previous
 * ones. Results go straight into the run images, which live in page-aligned
 * host memory that the GPU shares on Apple silicon.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cuMetal.h"

#include "GDALImage.h"
#include "cuAmpcorParameter.h"
#include "cuArrays.h"
#include "cuSincOverSampler.h"
#include "float2.h"

#include <isce3/matchtemplate/pycuampcor/cuMetalSource.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <cmath>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace isce3::matchtemplate::pycuampcor {

namespace {

// Kernel parameter structs; layouts match cuAmpcor.metal
struct GatherParams { int inNX, inNY, outNX, outNY, absolute, withMagnitude; };
struct Shape2 { int inNX, inNY, outNX, outNY, offsetX, offsetY; };
struct PackParams { int tNX, tNY, iNX, iNY, outNX, outNY; };
struct InsertParams { int inNX, inNY, outNY, offsetX, offsetY, elemWords; };
struct VarParams { int NX, NY, templateSize; };
struct SatParams { int nx, ny; };
struct NormParams { int corNX, corNY, refNX, refNY, secNX, secNY; };
struct TimeCorrParams { int tNX, tNY, iNX, iNY, rNX, rNY; };
struct FFTParams {
    int n, nradix, sign, linesPerImage, imageSize, lineStride, elemStride, lines, group;
    int nx, ny, loadMode, storeMode;
    int srcNX, srcNY, src2NX, src2NY, dstNX, dstNY, validRows;
    float coef;
};
struct DerampParams { int nx, ny, axis; };
struct ExtractOffsetParams { int xOldRange, yOldRange, xNewRange, yNewRange, count; };
struct SubPixelParams { int ovsZoomIn, ovsRaw, xHalfRange, yHalfRange, count; };
struct SincParams {
    int inNX, inNY, outNX, outNY;
    int factor, covs, decfactor, intplength;
    int startX, startY, size;
};

constexpr int REDUCE_THREADS = 256;   // must match cuAmpcor.metal
constexpr int MAX_FFT_LENGTH = 2048;  // 2n complex in 32 KB threadgroup memory
constexpr int MAX_FFT_RADIX = 31;     // largest radix with a butterfly kernel

// FFT factors: radix 4, 2, then odd factors (radix 4 first: fewer stages,
// i.e. fewer threadgroup memory passes)
std::vector<int> fftRadices(int n)
{
    std::vector<int> radices;
    int m = n;
    while (m % 4 == 0) { radices.push_back(4); m /= 4; }
    while (m % 2 == 0) { radices.push_back(2); m /= 2; }
    for (int f = 3; m > 1; f += 2)
        while (m % f == 0) { radices.push_back(f); m /= f; }
    return radices;
}

// whether fft1d can transform length n: fits threadgroup memory and every
// prime factor has a butterfly
bool fftSupported(int n)
{
    if (n < 1 || n > MAX_FFT_LENGTH) return false;
    for (int r : fftRadices(n))
        if (r > MAX_FFT_RADIX) return false;
    return true;
}

// Relative cost of a 2D FFT of n x n: complex operations per output of each
// stage (radix-4/2 butterflies, general radix r: r - 1 twiddles + (r - 1)^2
// DFT terms per r outputs) plus a load/store pass per stage
double fftCost(int n)
{
    double c = 0;
    for (int r : fftRadices(n))
        c += 1.0 + (r == 4 ? 1.0 : r == 2 ? 0.75 : ((r - 1) + (r - 1.0) * (r - 1)) / r);
    return (double)n * n * c;
}

// Cheapest supported FFT length in [n, 1.5 n]. Zero padding a correlation
// beyond the search window size keeps the full-overlap lags exact: they
// never wrap around, since the template is not larger than the window.
// Returns -1 if no length in the range is supported.
int correlationLength(int n)
{
    int best = -1;
    for (int m = n; m <= n + n / 2; m++)
        if (fftSupported(m) && (best < 0 || fftCost(m) < fftCost(best))) best = m;
    return best;
}

struct FFTPlan {
    int n = 0;
    std::vector<int> radices;
    id<MTLBuffer> twiddles = nil;     // n roots (cos, sin)(2 pi k / n), shared by all stages
    id<MTLBuffer> radixBuffer = nil;  // radices as int, read by fft1d
};

// Process-wide Metal state: device, queue, kernel library and caches of
// pipelines and FFT plans. Several GPU feeding threads share it, so the
// caches are guarded by a mutex; returned entries are never removed.
class Context {
public:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    // initialized once, thread-safe; nullptr when no Metal device exists
    static Context *get()
    {
        static Context ctx;
        static bool ok = false;
        static std::once_flag once;
        std::call_once(once, [] { ok = ctx.init(); });
        return ok ? &ctx : nullptr;
    }

    // compute pipeline of kernel `name`, built on first use
    id<MTLComputePipelineState> pipeline(const std::string &name)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;
        id<MTLFunction> f = [library newFunctionWithName:@(name.c_str())];
        if (!f) throw std::runtime_error("no Metal kernel " + name);
        NSError *err = nil;
        id<MTLComputePipelineState> p =
            [device newComputePipelineStateWithFunction:f error:&err];
        if (!p)
            throw std::runtime_error("Metal pipeline " + name + ": " +
                                     err.localizedDescription.UTF8String);
        pipelines[name] = p;
        return p;
    }

    // Mixed-radix plan of length n (radices 4, 2, then odd factors)
    const FFTPlan &fftPlan(int n)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = plans.find(n);
        if (it != plans.end()) return it->second;
        if (!fftSupported(n))
            throw std::runtime_error("unsupported Metal FFT length " + std::to_string(n));
        FFTPlan plan;
        plan.n = n;
        plan.radices = fftRadices(n);
        std::vector<float2> tw(n);
        for (int k = 0; k < n; k++) {
            const double a = 2.0 * M_PI * k / n;
            tw[k] = make_float2((float)std::cos(a), (float)std::sin(a));
        }
        plan.twiddles = [device newBufferWithBytes:tw.data() length:n * sizeof(float2)
                                           options:MTLResourceStorageModeShared];
        plan.radixBuffer = [device newBufferWithBytes:plan.radices.data()
                                               length:plan.radices.size() * sizeof(int)
                                              options:MTLResourceStorageModeShared];
        return plans[n] = plan;
    }

private:
    id<MTLLibrary> library = nil;
    std::mutex mutex;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
    std::map<int, FFTPlan> plans;

    // kernels are compiled from the source embedded by CMake (cuMetalSource.h)
    bool init()
    {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) return false;
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.mathMode = MTLMathModeSafe;  // follow the CPU arithmetic
            NSError *error = nil;
            library = [device newLibraryWithSource:@(cuAmpcorMetalSource)
                                           options:options error:&error];
            if (!library)
                throw std::runtime_error(std::string("Metal ampcor kernels: ") +
                                         error.localizedDescription.UTF8String);
            queue = [device newCommandQueue];
            return true;
        }
    }
};

size_t roundToPage(size_t bytes)
{
    const size_t page = getpagesize();
    return std::max<size_t>((bytes + page - 1) / page * page, page);
}

// Zero-copy buffer over page-aligned host memory (pageAlignedAlloc). The
// length is rounded up to whole pages, which pageAlignedAlloc also
// allocates; the host keeps ownership (no deallocator) and must outlive
// the buffer.
id<MTLBuffer> wrap(const void *data, size_t bytes)
{
    id<MTLBuffer> b = [Context::get()->device
        newBufferWithBytesNoCopy:const_cast<void *>(data) length:roundToPage(bytes)
                         options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) throw std::runtime_error("Metal buffer over host memory failed");
    return b;
}

// Batch of `count` images of height x width elements of T in a GPU buffer
template <typename T>
struct Batch {
    id<MTLBuffer> buffer = nil;
    int height = 0, width = 0, count = 0;

    Batch() = default;
    Batch(int h, int w, int n) : height(h), width(w), count(n)
    {
        // shared storage: the CPU reads/writes data() directly; at least
        // 4 bytes since Metal rejects empty buffers
        buffer = [Context::get()->device newBufferWithLength:std::max<size_t>(bytes(), 4)
                                                     options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
    }
    size_t size() const { return (size_t)height * width; }
    size_t elements() const { return size() * count; }
    size_t bytes() const { return elements() * sizeof(T); }
    T *data() const { return (T *)buffer.contents; }
};

// Per-kernel GPU time (ISCE3_METAL_PROFILE=1): every kernel runs in its own
// command buffer; totals are printed at the end of runAmpcorMetal.
// Not synchronized: meant for runs with one GPU feeding thread.
struct Profile {
    bool on = std::getenv("ISCE3_METAL_PROFILE") != nullptr;
    std::map<std::string, double> gpu;
    double cpuLoad = 0;
};
Profile profile;

// Compute command encoder with dispatch helpers. Usage:
// e.kernel(name).buf(...).bytes(...).grid(...); buffer indices follow the
// call order of buf/bytes. The default (serial) compute encoder runs the
// dispatches in order, so no barriers are needed between kernels.
class Encoder {
public:
    explicit Encoder(id<MTLCommandBuffer> c) : cmd(c), enc([c computeCommandEncoder]) {}
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;

    // end encoding; returns the command buffer to commit
    id<MTLCommandBuffer> finish()
    {
        if (profile.on) flushProfile();
        [enc endEncoding];
        return cmd;
    }

    Encoder &kernel(const char *name, const std::string &label = "")
    {
        if (profile.on) {
            flushProfile();
            current = label.empty() ? name : label;
        }
        pso = Context::get()->pipeline(name);
        [enc setComputePipelineState:pso];
        nbuf = 0;
        return *this;
    }
    template <typename T>
    Encoder &buf(const Batch<T> &b) { [enc setBuffer:b.buffer offset:0 atIndex:nbuf++]; return *this; }
    Encoder &buf(id<MTLBuffer> b) { [enc setBuffer:b offset:0 atIndex:nbuf++]; return *this; }
    template <typename P>
    Encoder &bytes(const P &p) { [enc setBytes:&p length:sizeof(P) atIndex:nbuf++]; return *this; }

    // one thread per (x, y, z); threadgroups are one SIMD width along x
    // (the contiguous image axis) and as many rows along y as fit
    void grid(int x, int y = 1, int z = 1)
    {
        if (x <= 0 || y <= 0 || z <= 0) return;
        const NSUInteger tw = std::min<NSUInteger>(pso.threadExecutionWidth, x);
        const NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(
            pso.maxTotalThreadsPerThreadgroup / tw, y));
        [enc dispatchThreads:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    }
    // one threadgroup of REDUCE_THREADS threads per image
    void groups(int n)
    {
        if (n <= 0) return;
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(REDUCE_THREADS, 1, 1)];
    }

private:
    id<MTLComputePipelineState> pso = nil;
    int nbuf = 0;
    std::string current;

    // profiling: run the kernels encoded since the last label alone and
    // add their GPU time; continue in a new command buffer
    void flushProfile()
    {
        if (current.empty()) return;
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        profile.gpu[current] += cmd.GPUEndTime - cmd.GPUStartTime;
        current.clear();
        cmd = [Context::get()->queue commandBuffer];
        enc = [cmd computeCommandEncoder];
    }
};

// -------------------------------------------------------------- operations

// Input of the first pass of fft2d instead of the batch itself (fft1d
// fftLoad): 1 copy of a complex batch c, 2 its spectrum padded to the batch
// size (padSpectrum, c of nx x ny), 3 real pair f + i f2 zero padded
// (packRealPair, f of nx x ny, f2 of nx2 x ny2), 5 conj(T) S * coef of the
// packed spectrum c of the batch size (mulConjPacked)
struct FFTLoad {
    int mode = 0;
    id<MTLBuffer> c = nil, f = nil, f2 = nil;
    int nx = 0, ny = 0, nx2 = 0, ny2 = 0;
    float coef = 1.0f;
};

// Output of the last pass of fft2d instead of the batch (fft1d fftStore):
// 1 magnitudes into f (complexAbs, same size), 2 real parts of the top-left
// nx x ny into f (extractReal)
struct FFTStore {
    int mode = 0;
    id<MTLBuffer> f = nil;
    int nx = 0, ny = 0;
};

// Unnormalized 2D DFT of every image, in place (sign -1 = FFTW_FORWARD).
// rows: rows transformed by the row pass (all by default); with
// rowsFirst = false, columns go first and only those rows of the result are
// valid; with rowsFirst, the other rows must be zero on input (or are taken
// as zero with a load). load/store replace the input of the first pass and
// the output of the last one.
void fft2d(Encoder &e, const Batch<float2> &b, int sign, int rows = -1, bool rowsFirst = true,
           const FFTLoad &load = {}, const FFTStore &store = {})
{
    auto pass = [&](int n, int lines, FFTParams p, bool first, bool last) {
        p.nx = b.height;
        p.ny = b.width;
        if (first && load.mode) {
            p.loadMode = load.mode;
            p.srcNX = load.nx; p.srcNY = load.ny;
            p.src2NX = load.nx2; p.src2NY = load.ny2;
            p.coef = load.coef;
        } else if (!first && load.mode && rowsFirst && rows < b.height) {
            // rows the first pass did not write are zero
            p.loadMode = 4;
            p.validRows = rows;
        }
        if (last && store.mode) {
            p.storeMode = store.mode;
            p.dstNX = store.nx; p.dstNY = store.ny;
        }
        const FFTPlan &plan = Context::get()->fftPlan(n);
        p.n = n;
        p.nradix = (int)plan.radices.size();
        p.sign = sign;
        p.lines = lines;
        // lines per threadgroup: 2 n group complex fit in 32 KB
        p.group = std::max(1, std::min(16, MAX_FFT_LENGTH / n));
        e.kernel("fft1d", "fft1d n=" + std::to_string(n) + (p.elemStride == 1 ? " rows" : " cols"))
            .buf(b).buf(plan.twiddles).buf(plan.radixBuffer).bytes(p)
            .buf(load.c ? load.c : b.buffer).buf(load.f ? load.f : b.buffer)
            .buf(load.f2 ? load.f2 : b.buffer).buf(store.f ? store.f : b.buffer);
        // Stockham ping-pong buffers; Metal needs a multiple of 16 bytes
        [e.enc setThreadgroupMemoryLength:(2 * n * p.group * sizeof(float2) + 15) / 16 * 16
                                  atIndex:0];
        [e.enc dispatchThreadgroups:MTLSizeMake((lines + p.group - 1) / p.group, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    const int nx = b.height, ny = b.width, size = nx * ny;
    if (rows < 0) rows = nx;
    // FFTParams {.., linesPerImage, imageSize, lineStride, elemStride, ..}:
    // a row is ny contiguous elements, a column ny-strided
    auto rowPass = [&](bool first) {
        pass(ny, b.count * rows, FFTParams{0, 0, 0, rows, size, ny, 1, 0, 0}, first, !first);
    };
    auto colPass = [&](bool first) {
        pass(nx, b.count * ny, FFTParams{0, 0, 0, ny, size, 1, ny, 0, 0}, first, !first);
    };
    if (rowsFirst) { rowPass(true); colPass(false); }
    else { colPass(true); rowPass(false); }
}

// cuArraysAbs
void complexAbs(Encoder &e, const Batch<float2> &in, const Batch<float> &out)
{
    e.kernel("complexAbs").buf(in).buf(out).grid((int)in.elements());
}

// cuArraysSubtractMean, in place
void subtractMean(Encoder &e, const Batch<float> &b)
{
    e.kernel("subtractMean").buf(b).bytes((int)b.size()).groups(b.count);
}

// cuFreqCorrelator / cuCorrTimeDomain: correlation of every template with
// its search window over the full-overlap lags (results: (iNX - tNX + 1) x
// (iNY - tNY + 1)); algorithm 0 = frequency domain
struct Correlator {
    int algorithm;
    // workT: packed spectrum FFT(t + i s); workS: conj(T) S, then its
    // inverse transform (both at the padded correlationLength size)
    Batch<float2> workT, workS;

    Correlator(int algorithm_, int nx, int ny, int count) : algorithm(algorithm_)
    {
        if (algorithm == 0) {
            const int fx = correlationLength(nx), fy = correlationLength(ny);
            workT = Batch<float2>(fx, fy, count);
            workS = Batch<float2>(fx, fy, count);
        }
    }

    void encode(Encoder &e, const Batch<float> &templates, const Batch<float> &images,
                const Batch<float> &results)
    {
        if (algorithm != 0) {
            e.kernel("corrTimeDomain").buf(templates).buf(images).buf(results)
                .bytes(TimeCorrParams{templates.height, templates.width, images.height,
                                      images.width, results.height, results.width})
                .grid(results.width, results.height, results.count);
            return;
        }
        // both real inputs through one complex FFT (t + i s) of the padded
        // size, packed by the first pass; rows beyond the inputs are zero
        const int nx = workT.height, ny = workT.width;
        FFTLoad pack;
        pack.mode = 3;
        pack.f = templates.buffer; pack.nx = templates.height; pack.ny = templates.width;
        pack.f2 = images.buffer; pack.nx2 = images.height; pack.ny2 = images.width;
        fft2d(e, workT, -1, std::max(templates.height, images.height), true, pack);
        // unnormalized forward and inverse transforms scale by nx * ny: the
        // result is the linear correlation, as cuFreqCorrelator for nx x ny
        // inverse of conj(T) S (computed by the first pass): columns first,
        // then only the result rows, whose real parts the last pass stores
        // into the results
        FFTLoad mulConj;
        mulConj.mode = 5;
        mulConj.c = workT.buffer;
        mulConj.coef = 1.0f / (nx * ny);
        FFTStore real;
        real.mode = 2;
        real.f = results.buffer; real.nx = results.height; real.ny = results.width;
        fft2d(e, workS, +1, results.height, false, mulConj, real);
    }
};

// cuNormalizeSAT: divides the correlation by
// sqrt(sum t^2 * (sum s^2 - (sum s)^2 / N)) over the template-sized box of
// the secondary at each lag; box sums come from summed-area tables (SATs)
struct Normalizer {
    Batch<float> refSum2, sat, sat2;  // per-image sum t^2, SATs of s and s^2

    Normalizer(int nx, int ny, int count)
        : refSum2(1, 1, count), sat(nx, ny, count), sat2(nx, ny, count) {}

    void encode(Encoder &e, const Batch<float> &corr, const Batch<float> &ref,
                const Batch<float> &sec)
    {
        e.kernel("sumSquare").buf(ref).buf(refSum2).bytes((int)ref.size()).groups(ref.count);
        const SatParams sp{sec.height, sec.width};
        e.kernel("satRows").buf(sec).buf(sat).buf(sat2).bytes(sp).bytes(ref.count);
        // one SIMD group (32 threads) per row of every image
        [e.enc dispatchThreadgroups:MTLSizeMake(sec.height * ref.count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        e.kernel("satCols").buf(sat).buf(sat2).bytes(sp).grid(sec.width, ref.count);
        e.kernel("normalizeSat").buf(corr).buf(refSum2).buf(sat).buf(sat2)
            .bytes(NormParams{corr.height, corr.width, ref.height, ref.width, sec.height, sec.width})
            .grid(corr.width, corr.height, corr.count);
    }
};

// GPU copy of `bytes` (a multiple of 4), ordered with the other kernels
void copyBuffer(Encoder &e, id<MTLBuffer> in, id<MTLBuffer> out, size_t bytes)
{
    e.kernel("copyWords").buf(in).buf(out).grid((int)(bytes / 4));
}

// Spectrum padding of cuArraysPaddingMany
void padSpectrum(Encoder &e, const Batch<float2> &in, const Batch<float2> &out)
{
    e.kernel("padSpectrum").buf(in).buf(out)
        .bytes(Shape2{in.height, in.width, out.height, out.width, 0, 0})
        .grid(out.width, out.height, in.count);
}

// cuOverSamplerC2C: forward (+1) of in, spectrum padding, inverse (-1) into out
struct OverSamplerC2C {
    Batch<float2> workIn;

    OverSamplerC2C(int nx, int ny, int count) : workIn(nx, ny, count) {}

    // oversampled in into out, or only its magnitudes into outAbs if given
    // (complexAbs fused into the last pass; out is then scratch)
    void encode(Encoder &e, const Batch<float2> &in, const Batch<float2> &out,
                const Batch<float> *outAbs = nullptr)
    {
        // forward FFT of a copy (the first pass reads in), leaving in unchanged
        FFTLoad copy;
        copy.mode = 1;
        copy.c = in.buffer;
        fft2d(e, workIn, +1, -1, true, copy);
        // inverse of the padded spectrum (the first pass pads)
        FFTLoad pad;
        pad.mode = 2;
        pad.c = workIn.buffer; pad.nx = workIn.height; pad.ny = workIn.width;
        FFTStore abs;
        if (outAbs) {
            abs.mode = 1;
            abs.f = outAbs->buffer;
        }
        fft2d(e, out, -1, -1, true, pad, abs);
    }
};

// cuOverSamplerR2R: real images oversampled through the spectrum
struct OverSamplerR2R {
    Batch<float2> workIn, workOut;

    OverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int count)
        : workIn(inNX, inNY, count), workOut(outNX, outNY, count) {}

    void encode(Encoder &e, const Batch<float> &in, const Batch<float> &out)
    {
        e.kernel("padRealToComplex").buf(in).buf(workIn)
            .bytes(Shape2{in.height, in.width, workIn.height, workIn.width, 0, 0})
            .grid(workIn.width, workIn.height, in.count);
        fft2d(e, workIn, +1);
        padSpectrum(e, workIn, workOut);
        fft2d(e, workOut, -1);
        e.kernel("extractReal").buf(workOut).buf(out)
            .bytes(Shape2{workOut.height, workOut.width, out.height, out.width, 0, 0})
            .grid(out.width, out.height, out.count);
    }
};

// Run images (full offset field) shared with the GPU; chunks write disjoint
// windows, so concurrent chunks need no synchronization
struct RunImages {
    id<MTLBuffer> offset, snr, cov, corr;
    int width;  // windows across of the run images
};

/**
 * Metal version of cuAmpcorChunk: same arrays and steps, one command
 * buffer per chunk. Array names follow cuAmpcorChunk (c/r prefix: complex /
 * real batches of nwd * nwa windows). A MetalChunk is reused for one chunk
 * at a time: submit() may only follow wait() of its previous chunk, since
 * load() overwrites buffers the GPU reads.
 */
class MetalChunk {
public:
    MetalChunk(cuAmpcorParameter *param_, GDALImage *reference_, GDALImage *secondary_,
               const RunImages &run_)
        : param(param_), reference(reference_), secondary(secondary_), run(run_),
          nwd(param->numberWindowDownInChunk), nwa(param->numberWindowAcrossInChunk),
          refChunk(param->maxReferenceChunkHeight, param->maxReferenceChunkWidth, 1),
          secChunk(param->maxSecondaryChunkHeight, param->maxSecondaryChunkWidth, 1),
          refOffDown(nwd, nwa, 1), refOffAcross(nwd, nwa, 1),
          secOffDown(nwd, nwa, 1), secOffAcross(nwd, nwa, 1),
          cRefRaw(param->windowSizeHeightRaw, param->windowSizeWidthRaw, nwd * nwa),
          cSecRaw(param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw, nwd * nwa),
          rRefRaw(cRefRaw.height, cRefRaw.width, cRefRaw.count),
          rSecRaw(cSecRaw.height, cSecRaw.width, cSecRaw.count),
          cSecZoomIn(param->searchWindowSizeHeightRawZoomIn, param->searchWindowSizeWidthRawZoomIn,
                     nwd * nwa),
          cRefOvs(param->windowSizeHeight, param->windowSizeWidth, nwd * nwa),
          cSecOvs(param->searchWindowSizeHeight, param->searchWindowSizeWidth, nwd * nwa),
          rRefOvs(cRefOvs.height, cRefOvs.width, cRefOvs.count),
          rSecOvs(cSecOvs.height, cSecOvs.width, cSecOvs.count),
          rCorrRaw(param->searchWindowSizeHeightRaw - param->windowSizeHeightRaw + 1,
                   param->searchWindowSizeWidthRaw - param->windowSizeWidthRaw + 1, nwd * nwa),
          rCorrZoomIn(param->searchWindowSizeHeight - param->windowSizeHeight + 1,
                      param->searchWindowSizeWidth - param->windowSizeWidth + 1, nwd * nwa),
          rCorrZoomInAdjust(param->searchWindowSizeHeight - param->windowSizeHeight,
                            param->searchWindowSizeWidth - param->windowSizeWidth, nwd * nwa),
          rCorrZoomInOvs(param->zoomWindowSize * param->oversamplingFactor,
                         param->zoomWindowSize * param->oversamplingFactor, nwd * nwa),
          offsetInit(nwd, nwa, 1), offsetZoomIn(nwd, nwa, 1), offsetFinal(nwd, nwa, 1),
          maxLocShift(nwd, nwa, 1), corrMaxValue(nwd, nwa, 1), rMaxval(nwd, nwa, 1),
          rCorrRawZoomIn(param->corrRawZoomInHeight, param->corrRawZoomInWidth, nwd * nwa),
          iCorrZoomInValid(param->corrRawZoomInHeight, param->corrRawZoomInWidth, nwd * nwa),
          rCorrSum(nwd, nwa, 1), iCorrValidCount(nwd, nwa, 1), rSnr(nwd, nwa, 1),
          rCov(nwd, nwa, 1), flowDirection(nwd, nwa, 1),
          corrRaw(param->algorithm, cSecRaw.height, cSecRaw.width, nwd * nwa),
          corrOvs(param->algorithm, cSecOvs.height, cSecOvs.width, nwd * nwa),
          normRaw(cSecRaw.height, cSecRaw.width, nwd * nwa),
          normOvs(cSecOvs.height, cSecOvs.width, nwd * nwa),
          ovsRef(cRefRaw.height, cRefRaw.width, nwd * nwa),
          ovsSec(cSecZoomIn.height, cSecZoomIn.width, nwd * nwa)
    {
        if (param->oversamplingMethod) {
            // sinc: the CPU sampler provides the filter table; only a
            // size x size window around the peak is computed (sincCols)
            sinc = std::make_unique<cuSincOverSamplerR2R>(param->oversamplingFactor);
            sincFilter = wrap(sinc->filter(), sinc->filterLength() * sizeof(float));
            const int size = 2 * sinc->sincWindow() * sinc->covs() + 1;
            sincRowsBuf = Batch<float>(rCorrZoomInAdjust.height, size, nwd * nwa);
            sincWindowBuf = Batch<float>(size, size, nwd * nwa);
            sincTapIndex = Batch<int>(2 * size, sinc->intplength(), nwd * nwa);
            sincTapCoef = Batch<float>(2 * size, sinc->intplength(), nwd * nwa);
            sincTapSum = Batch<float>(2, size, nwd * nwa);
        } else {
            ovsCorr = std::make_unique<OverSamplerR2R>(
                param->zoomWindowSize, param->zoomWindowSize,
                rCorrZoomInOvs.height, rCorrZoomInOvs.width, nwd * nwa);
        }
    }

    // Prepare chunk (idxDown, idxAcross) on the CPU and submit it to the GPU
    void submit(int idxDown, int idxAcross)
    {
        @autoreleasepool {
            setIndex(idxDown, idxAcross);
            const double t0 = CFAbsoluteTimeGetCurrent();
            load();
            profile.cpuLoad += CFAbsoluteTimeGetCurrent() - t0;
            Encoder e([Context::get()->queue commandBuffer]);
            encode(e);
            cmd = e.finish();
            [cmd commit];
        }
    }

    // Wait for the last submitted chunk
    void wait()
    {
        if (!cmd) return;
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError)
            throw std::runtime_error(std::string("Metal ampcor chunk: ") +
                                     cmd.error.localizedDescription.UTF8String);
        cmd = nil;
    }

private:
    cuAmpcorParameter *param;
    GDALImage *reference, *secondary;
    RunImages run;
    int nwd, nwa;  // windows per chunk (down, across)
    int idxChunkDown = 0, idxChunkAcross = 0, idxChunk = 0;
    int nWindowsDown = 0, nWindowsAcross = 0;
    id<MTLCommandBuffer> cmd = nil;

    Batch<float2> refChunk, secChunk;
    Batch<int> refOffDown, refOffAcross, secOffDown, secOffAcross;
    Batch<float2> cRefRaw, cSecRaw;
    Batch<float> rRefRaw, rSecRaw;
    Batch<float2> cSecZoomIn, cRefOvs, cSecOvs;
    Batch<float> rRefOvs, rSecOvs;
    Batch<float> rCorrRaw, rCorrZoomIn, rCorrZoomInAdjust, rCorrZoomInOvs;
    Batch<int2> offsetInit, offsetZoomIn;
    Batch<float2> offsetFinal;
    Batch<int2> maxLocShift;
    Batch<float> corrMaxValue, rMaxval, rCorrRawZoomIn;
    Batch<int> iCorrZoomInValid;
    Batch<float> rCorrSum;
    Batch<int> iCorrValidCount;
    Batch<float> rSnr;
    Batch<float3> rCov;
    Batch<float2> flowDirection;
    Correlator corrRaw, corrOvs;
    Normalizer normRaw, normOvs;
    OverSamplerC2C ovsRef, ovsSec;
    std::unique_ptr<cuSincOverSamplerR2R> sinc;
    id<MTLBuffer> sincFilter = nil;
    Batch<float> sincRowsBuf;    // y-interpolated rows of the separable sinc
    Batch<float> sincWindowBuf;  // oversampled window around the peak
    Batch<int> sincTapIndex;     // sinc taps of every window coordinate
    Batch<float> sincTapCoef, sincTapSum;  // tap coefficients, their sum per coordinate
    std::unique_ptr<OverSamplerR2R> ovsCorr;

    // cuAmpcorChunk::setIndex
    void setIndex(int idxDown, int idxAcross)
    {
        idxChunkDown = idxDown;
        idxChunkAcross = idxAcross;
        idxChunk = idxAcross + idxDown * param->numberChunkAcross;
        nWindowsDown = (idxDown == param->numberChunkDown - 1)
            ? param->numberWindowDown - nwd * (param->numberChunkDown - 1) : nwd;
        nWindowsAcross = (idxAcross == param->numberChunkAcross - 1)
            ? param->numberWindowAcross - nwa * (param->numberChunkAcross - 1) : nwa;
    }

    // global window index of window (i, j) of the chunk (cuAmpcorChunk::getRelativeOffset)
    int windowIndex(int i, int j) const
    {
        const int iDown = std::min(i, nWindowsDown - 1);
        const int iAcross = std::min(j, nWindowsAcross - 1);
        return (iDown + idxChunkDown * nwd) * param->numberWindowAcross +
               idxChunkAcross * nwa + iAcross;
    }

    // Window offsets within the chunks, flow directions and chunk data (CPU),
    // written straight into the shared buffers. Windows past the last valid
    // one of an edge chunk repeat it (windowIndex clamps).
    void load()
    {
        const bool dlc = !param->flowDirectionDown.empty();
        for (int i = 0; i < nwd; i++) {
            for (int j = 0; j < nwa; j++) {
                const int k = i * nwa + j, w = windowIndex(i, j);
                refOffDown.data()[k] = param->referenceStartPixelDown[w] -
                                       param->referenceChunkStartPixelDown[idxChunk];
                refOffAcross.data()[k] = param->referenceStartPixelAcross[w] -
                                         param->referenceChunkStartPixelAcross[idxChunk];
                secOffDown.data()[k] = param->secondaryStartPixelDown[w] -
                                       param->secondaryChunkStartPixelDown[idxChunk];
                secOffAcross.data()[k] = param->secondaryStartPixelAcross[w] -
                                         param->secondaryChunkStartPixelAcross[idxChunk];
                if (dlc)
                    flowDirection.data()[k] = param->flowDirection(w);
            }
        }
        if (!reference->isComplex() || !secondary->isComplex())
            throw std::invalid_argument("real images not supported");
        reference->loadToDevice(refChunk.data(),
            param->referenceChunkStartPixelDown[idxChunk],
            param->referenceChunkStartPixelAcross[idxChunk],
            param->referenceChunkHeight[idxChunk], param->referenceChunkWidth[idxChunk]);
        secondary->loadToDevice(secChunk.data(),
            param->secondaryChunkStartPixelDown[idxChunk],
            param->secondaryChunkStartPixelAcross[idxChunk],
            param->secondaryChunkHeight[idxChunk], param->secondaryChunkWidth[idxChunk]);
    }

    // windows of a chunk (row length lda) into a batch; amplitudes only
    // without deramping (derampMethod 0), as the CPU
    // and, if given, their magnitudes into magnitude (complexAbs fused)
    void gather(Encoder &e, const Batch<float2> &chunk, int lda, const Batch<int> &offDown,
                const Batch<int> &offAcross, const Batch<float2> &out,
                const Batch<float> *magnitude = nullptr)
    {
        e.kernel("gatherBatch").buf(chunk).buf(out).buf(offDown).buf(offAcross)
            .bytes(GatherParams{chunk.height, lda, out.height, out.width,
                                param->derampMethod == 0, magnitude != nullptr})
            .buf(magnitude ? magnitude->buffer : out.buffer)
            .grid(out.width, out.height, out.count);
    }

    // cuDeramp, in place (derampMethod 1 only)
    void deramp(Encoder &e, const Batch<float2> &b)
    {
        if (param->derampMethod != 1) return;
        e.kernel("deramp").buf(b).bytes(DerampParams{b.height, b.width, param->derampAxis})
            .groups(b.count);
    }

    // cuArraysMaxloc2D: peak location and value of every image
    void maxloc(Encoder &e, const Batch<float> &images, const Batch<int2> &loc,
                const Batch<float> &val)
    {
        const int shape[2] = {images.height, images.width};
        e.kernel("maxloc2D").buf(images).buf(loc).buf(val);
        [e.enc setBytes:shape length:sizeof(shape) atIndex:3];
        e.groups(images.count);
    }

    // chunk result (nwd x nwa elements of `words` 32-bit words) into a run
    // image; full chunks also at the edges: the run images are sized to whole
    // chunks (cuAmpcorController), as for cuAmpcorChunk
    void insert(Encoder &e, id<MTLBuffer> in, id<MTLBuffer> out, int words)
    {
        e.kernel("insertChunk").buf(in).buf(out)
            .bytes(InsertParams{nwd, nwa, run.width, idxChunkDown * nwd,
                                idxChunkAcross * nwa, words})
            .grid(nwa, nwd);
    }

    // cuAmpcorChunk::run
    void encode(Encoder &e)
    {
        const int n = nwd * nwa;
        // reference windows: amplitudes with the mean removed
        gather(e, refChunk, param->referenceChunkWidth[idxChunk], refOffDown, refOffAcross,
               cRefRaw, &rRefRaw);
        subtractMean(e, rRefRaw);
        // secondary search windows
        gather(e, secChunk, param->secondaryChunkWidth[idxChunk], secOffDown, secOffAcross,
               cSecRaw, &rSecRaw);
        // correlation before oversampling and its integer peak
        corrRaw.encode(e, rRefRaw, rSecRaw, rCorrRaw);
        normRaw.encode(e, rCorrRaw, rRefRaw, rSecRaw);
        if (param->flowDirectionDown.empty()) {
            maxloc(e, rCorrRaw, offsetInit, rMaxval);
        } else {
            // DLC: peak constrained to the flow line (or to a band around it)
            const int shape[2] = {rCorrRaw.height, rCorrRaw.width};
            e.kernel(param->flowBandHalfWidth.empty() ? "maxlocDLC" : "maxlocBand").buf(rCorrRaw).buf(flowDirection).buf(offsetInit).buf(rMaxval);
            [e.enc setBytes:shape length:sizeof(shape) atIndex:4];
            [e.enc setBytes:&n length:sizeof(n) atIndex:5];
            e.grid(n);
        }
        // covariance and SNR
        e.kernel("estimateVariance").buf(rCorrRaw).buf(offsetInit).buf(rMaxval).buf(rCov)
            .bytes(VarParams{rCorrRaw.height, rCorrRaw.width, (int)rRefRaw.size()}).grid(n);
        e.kernel("extractCorr").buf(rCorrRaw).buf(rCorrRawZoomIn).buf(iCorrZoomInValid).buf(offsetInit)
            .bytes(Shape2{rCorrRaw.height, rCorrRaw.width, rCorrRawZoomIn.height, rCorrRawZoomIn.width, 0, 0})
            .grid(rCorrRawZoomIn.width, rCorrRawZoomIn.height, n);
        e.kernel("sumCorr").buf(rCorrRawZoomIn).buf(iCorrZoomInValid).buf(rCorrSum).buf(iCorrValidCount)
            .bytes((int)rCorrRawZoomIn.size()).groups(n);
        e.kernel("estimateSnr").buf(rCorrSum).buf(iCorrValidCount).buf(rMaxval).buf(rSnr).grid(n);
        // secondary extraction around the peak for the zoomed-in search:
        // offsetInit becomes the extraction start, maxLocShift the shift of
        // the peak from the zoom window center where it hits the edge
        e.kernel("secondaryExtractOffset").buf(offsetInit).buf(maxLocShift)
            .bytes(ExtractOffsetParams{param->halfSearchRangeDownRaw, param->halfSearchRangeAcrossRaw,
                                       param->halfZoomWindowSizeRaw, param->halfZoomWindowSizeRaw, n})
            .grid(n);
        // oversampled reference
        deramp(e, cRefRaw);
        ovsRef.encode(e, cRefRaw, cRefOvs, &rRefOvs);
        subtractMean(e, rRefOvs);
        // oversampled secondary
        e.kernel("extractComplexOffsets").buf(cSecRaw).buf(cSecZoomIn).buf(offsetInit)
            .bytes(Shape2{cSecRaw.height, cSecRaw.width, cSecZoomIn.height, cSecZoomIn.width, 0, 0})
            .grid(cSecZoomIn.width, cSecZoomIn.height, n);
        deramp(e, cSecZoomIn);
        ovsSec.encode(e, cSecZoomIn, cSecOvs, &rSecOvs);
        // oversampled correlation
        corrOvs.encode(e, rRefOvs, rSecOvs, rCorrZoomIn);
        normOvs.encode(e, rCorrZoomIn, rRefOvs, rSecOvs);
        e.kernel("extractFloat").buf(rCorrZoomIn).buf(rCorrZoomInAdjust)
            .bytes(Shape2{rCorrZoomIn.height, rCorrZoomIn.width,
                          rCorrZoomInAdjust.height, rCorrZoomInAdjust.width, 0, 0})
            .grid(rCorrZoomInAdjust.width, rCorrZoomInAdjust.height, n);
        // correlation surface oversampling and its peak
        if (param->oversamplingMethod) {
            // separable sinc: taps per coordinate (both axes), then y and x
            // passes over a (2 range + 1)^2 window centered on the shifted peak
            const int range = sinc->sincWindow() * sinc->covs();
            const SincParams sp{rCorrZoomInAdjust.height, rCorrZoomInAdjust.width,
                                rCorrZoomInOvs.height, rCorrZoomInOvs.width,
                                param->oversamplingFactor * param->rawDataOversamplingFactor,
                                sinc->covs(), sinc->decfactor(), sinc->intplength(),
                                rCorrZoomInOvs.height / 2 - range, rCorrZoomInOvs.width / 2 - range,
                                2 * range + 1};
            e.kernel("sincTaps").buf(maxLocShift).buf(sincFilter).buf(sincTapIndex)
                .buf(sincTapCoef).buf(sincTapSum).bytes(sp).grid(sp.size, 2, n);
            e.kernel("sincRows").buf(rCorrZoomInAdjust).buf(sincRowsBuf).buf(sincTapIndex)
                .buf(sincTapCoef).bytes(sp).grid(sp.size, sp.inNX, n);
            // only the window around the peak, compact; the surface is 0 elsewhere
            e.kernel("sincCols").buf(sincRowsBuf).buf(sincWindowBuf).buf(sincTapIndex)
                .buf(sincTapCoef).buf(sincTapSum).bytes(sp).grid(sp.size, sp.size, n);
            e.kernel("maxlocSinc").buf(sincWindowBuf).buf(maxLocShift).buf(offsetZoomIn)
                .buf(corrMaxValue).bytes(sp).groups(n);
        } else {
            ovsCorr->encode(e, rCorrZoomInAdjust, rCorrZoomInOvs);
            maxloc(e, rCorrZoomInOvs, offsetZoomIn, corrMaxValue);
        }
        e.kernel("subPixelOffset").buf(offsetInit).buf(offsetZoomIn).buf(offsetFinal)
            .bytes(SubPixelParams{param->oversamplingFactor, param->rawDataOversamplingFactor,
                                  param->halfSearchRangeDownRaw, param->halfSearchRangeAcrossRaw, n})
            .grid(n);
        // results into the run images
        insert(e, offsetFinal.buffer, run.offset, 2);
        insert(e, rSnr.buffer, run.snr, 1);
        insert(e, rCov.buffer, run.cov, 3);
        insert(e, corrMaxValue.buffer, run.corr, 1);
    }
};

} // namespace

// first call initializes the Metal context (compiles the kernels)
bool metalAvailable()
{
    return Context::get() != nullptr;
}

bool metalSupported(const cuAmpcorParameter *param)
{
    if (!metalAvailable()) return false;
    // FFT lengths of the correlators and oversamplers
    std::vector<int> lengths;
    if (param->algorithm == 0)
        for (int n : {param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
                      param->searchWindowSizeHeight, param->searchWindowSizeWidth})
            lengths.push_back(correlationLength(n));
    lengths.insert(lengths.end(), {param->windowSizeHeightRaw, param->windowSizeWidthRaw,
        param->windowSizeHeight, param->windowSizeWidth,
        param->searchWindowSizeHeightRawZoomIn, param->searchWindowSizeWidthRawZoomIn,
        param->searchWindowSizeHeight, param->searchWindowSizeWidth});
    if (!param->oversamplingMethod)
        lengths.insert(lengths.end(), {param->zoomWindowSize,
            param->zoomWindowSize * param->oversamplingFactor});
    for (int n : lengths) {
        if (!fftSupported(n)) {
            std::cout << "Metal FFT does not support length " << n << " (prime factors <= "
                      << MAX_FFT_RADIX << ", length <= " << MAX_FFT_LENGTH
                      << "); running ampcor on the CPU" << std::endl;
            return false;
        }
    }
    return true;
}

int runAmpcorMetal(const std::vector<MetalLayer> &layers, GDALImage *reference,
    GDALImage *secondary, const std::function<std::pair<int, int>()> &nextChunk,
    const std::function<void()> &chunkDone)
{
    // May run on several threads at once, each with its own slots; nextChunk
    // is shared with the CPU workers (hybrid scheduling)
    static_assert(sizeof(float3) == 12, "run cov image must be packed float3");

    // chunks in flight: the CPU loads one while the GPU runs the others;
    // each layer has its own slots (sized for its windows), made on first use
    const int nSlots = 4;
    struct Slots {
        RunImages run;
        std::vector<std::unique_ptr<MetalChunk>> chunks;
        std::vector<bool> busy;
    };
    std::vector<Slots> slots;
    for (const auto &l : layers)
        slots.push_back({RunImages{wrap(l.offsetImageRun->devData, l.offsetImageRun->getByteSize()),
                                   wrap(l.snrImageRun->devData, l.snrImageRun->getByteSize()),
                                   wrap(l.covImageRun->devData, l.covImageRun->getByteSize()),
                                   wrap(l.corrImageRun->devData, l.corrImageRun->getByteSize()),
                                   l.offsetImageRun->width},
                         {}, std::vector<bool>(nSlots, false)});

    // in-flight chunks oldest first: (layer, slot)
    std::deque<std::pair<int, int>> inFlight;
    auto retire = [&]() {
        const auto [l, s] = inFlight.front();
        inFlight.pop_front();
        slots[l].chunks[s]->wait();
        slots[l].busy[s] = false;
        chunkDone();
    };
    int processed = 0;
    for (auto item = nextChunk(); item.first >= 0; item = nextChunk(), processed++) {
        if (static_cast<int>(inFlight.size()) >= nSlots)
            retire();
        const int l = item.first;
        cuAmpcorParameter *param = layers[l].param;
        auto &ls = slots[l];
        if (ls.chunks.empty())
            for (int s = 0; s < nSlots; s++)
                ls.chunks.push_back(std::make_unique<MetalChunk>(param, reference,
                                                                 secondary, ls.run));
        // a free slot of the layer: fewer than nSlots chunks are in flight
        const int s = static_cast<int>(std::find(ls.busy.begin(), ls.busy.end(), false) -
                                       ls.busy.begin());
        ls.chunks[s]->submit(item.second / param->numberChunkAcross,
                             item.second % param->numberChunkAcross);
        ls.busy[s] = true;
        inFlight.push_back({l, s});
    }
    while (!inFlight.empty())
        retire();

    if (profile.on) {
        double total = 0;
        for (auto &kv : profile.gpu) total += kv.second;
        std::cout << "Metal ampcor profile: GPU " << total << " s, CPU chunk loading "
                  << profile.cpuLoad << " s" << std::endl;
        std::vector<std::pair<double, std::string>> sorted;
        for (auto &kv : profile.gpu) sorted.push_back({kv.second, kv.first});
        std::sort(sorted.rbegin(), sorted.rend());
        for (auto &kv : sorted)
            std::cout << "  " << kv.second << ": " << kv.first << " s ("
                      << 100 * kv.first / total << "%)" << std::endl;
        profile.gpu.clear();
        profile.cpuLoad = 0;
    }
    return processed;
}

} // namespace isce3::matchtemplate::pycuampcor
