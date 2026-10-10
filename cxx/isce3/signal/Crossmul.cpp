#include "Crossmul.h"

#include "Filter.h"
#include "Looks.h"
#include "Signal.h"

#include <algorithm>
#include <future>
#include <mutex>
#include <numeric>

/**
 * Compute the frequency response due to a subpixel shift introduced by
 * upsampling and downsampling

 * @param[in] oversample upsampling factor
 * @param[in] fft_size fft length in range direction
 * @param[in] blockRows number of rows of the block of data
 * @param[out] shiftImpact frequency response (a linear phase) to a sub-pixel
 * shift in time domain introduced by upsampling followed by downsampling
 */
void lookdownShiftImpact(size_t oversample, size_t fft_size, size_t blockRows,
        std::valarray<std::complex<float>> &shiftImpact)
{
    // range frequencies given fft_size and oversampling factor
    std::valarray<double> rangeFrequencies(oversample*fft_size);

    // sampling interval in range
    double dt = 1.0/oversample;

    // get the vector of range frequencies
    isce3::signal::fftfreq(dt, rangeFrequencies);

    // in the process of upsampling the SLCs, creating upsampled interferogram
    // and then looking down the upsampled interferogram to the original size of
    // the SLCs, a shift is introduced in range direction.
    // As an example for a signal with length of 5 and :
    // original sample locations:   0       1       2       3        4
    // upsampled sample locations:  0   0.5 1  1.5  2  2.5  3   3.5  4   4.5
    // Looked dow sample locations:   0.25    1.25    2.25    3.25    4.25
    // Obviously the signal after looking down would be shifted by 0.25 pixel in
    // range comared to the original signal. Since a shift in time domain introduces
    // a linear phase in frequency domain, we compute the impact in frequency domain.

    // the constant shift based on the oversampling factor
    double shift = 0.0;
    shift = (1.0 - 1.0/oversample)/2.0;

    // compute the frequency response of the subpixel shift in range direction
    std::valarray<std::complex<float>> shiftImpactLine(oversample*fft_size);
    for (size_t col=0; col<shiftImpactLine.size(); ++col) {
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<float> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The impact is the same for each range line. Therefore copying the line
    // for the block
    for (size_t line = 0; line < blockRows; ++line) {
        shiftImpact[std::slice(line*fft_size*oversample, fft_size*oversample, 1)] = shiftImpactLine;
    }
}

// Utility function to get number of OpenMP threads
// (gcc sometimes has problems with omp_get_num_threads)
size_t omp_thread_count() {
    size_t n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

void isce3::signal::Crossmul::
crossmul(isce3::io::Raster& refSlcRaster,
        isce3::io::Raster& secSlcRaster,
        isce3::io::Raster& ifgRaster,
        isce3::io::Raster& coherenceRaster,
        isce3::io::Raster* rngOffsetRaster) const
{
    _crossmul(refSlcRaster, secSlcRaster,
              {{&ifgRaster, &coherenceRaster, _rangeLooks, _azimuthLooks}},
              rngOffsetRaster);
}

void isce3::signal::Crossmul::
crossmul(isce3::io::Raster& refSlcRaster,
        isce3::io::Raster& secSlcRaster,
        isce3::io::Raster& ifgRaster,
        isce3::io::Raster& coherenceRaster,
        isce3::io::Raster& ifgRaster2,
        isce3::io::Raster& coherenceRaster2,
        int rangeLooks2, int azimuthLooks2,
        isce3::io::Raster* rngOffsetRaster) const
{
    if (rangeLooks2 < 1 || azimuthLooks2 < 1)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "crossmul multilook < 1");
    _crossmul(refSlcRaster, secSlcRaster,
              {{&ifgRaster, &coherenceRaster, _rangeLooks, _azimuthLooks},
               {&ifgRaster2, &coherenceRaster2, rangeLooks2, azimuthLooks2}},
              rngOffsetRaster);
}

void isce3::signal::Crossmul::
_crossmul(isce3::io::Raster& refSlcRaster,
        isce3::io::Raster& secSlcRaster,
        const std::vector<LooksOutput>& outputs,
        isce3::io::Raster* rngOffsetRaster) const
{
    // check consistency of input/output raster shapes
    size_t nrows = refSlcRaster.length();
    size_t ncols = refSlcRaster.width();

    // Making sure that the number of rows in each block (linesPerBlock)
    // to be an integer multiple of the number of azimuth looks of every
    // output (of their least common multiple)
    size_t azimuthLooksLcm = 1;
    for (const auto& out : outputs) {
        auto& ifgRaster = *out.ifg;
        auto& coherenceRaster = *out.coherence;
        if (ifgRaster.length() != coherenceRaster.length())
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "interferogram and coherence rasters length do not match");

        if (ifgRaster.width() != coherenceRaster.width())
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "interferogram and coherence rasters width do not match");

        // checking only multilook interferogram shape is sufficient
        // interferogram and coherence shapes checked to match above
        const auto output_rows = ifgRaster.length();
        const auto output_cols = ifgRaster.width();
        if (output_rows != nrows / out.azimuthLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "interferogram/coherence raster lengths of unexpected size");

        if (output_cols != ncols / out.rangeLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "interferogram/coherence raster widths of unexpected size");

        azimuthLooksLcm = std::lcm(azimuthLooksLcm,
                                   static_cast<size_t>(out.azimuthLooks));
    }
    const size_t linesPerBlock = std::max<size_t>(1, _linesPerBlock / azimuthLooksLcm)
                                 * azimuthLooksLcm;

    size_t nthreads = omp_thread_count();

    // Set flatten flag based range offset raster ptr value
    bool flatten = rngOffsetRaster ? true : false;

    //signal object for refSlc
    isce3::signal::Signal<float> refSignal(nthreads);

    //signal object for secSlc
    isce3::signal::Signal<float> secSignal(nthreads);


    // Compute FFT size (power of 2)
    size_t fft_size;
    refSignal.nextPowerOfTwo(ncols, fft_size);

    if (fft_size > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "fft_size > INT_MAX");
    if (_oversampleFactor * fft_size > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "_oversampleFactor * fft_size > INT_MAX");

    // number of blocks to process
    size_t nblocks = nrows / linesPerBlock;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * linesPerBlock) != 0) {
        nblocks += 1;
    }

    // size of not-unsampled valarray
    const auto spectrumSize = fft_size * linesPerBlock;

    // size of unsampled valarray
    const auto spectrumUpsampleSize = _oversampleFactor * spectrumSize;

    // storage for a block of reference SLC data
    std::valarray<std::complex<float>> refSlc(spectrumSize);

    // storage for a block of secondary SLC data
    std::valarray<std::complex<float>> secSlc(spectrumSize);

    // storage for a block of range offsets
    std::valarray<double> rngOffset(ncols*linesPerBlock);

    // next block of the reference and secondary SLCs and range offsets,
    // read while the current one is processed
    std::valarray<std::complex<float>> nextRefSlc(spectrumSize);
    std::valarray<std::complex<float>> nextSecSlc(spectrumSize);
    std::valarray<double> nextRngOffset(ncols*linesPerBlock);

    // storage for a simulated interferogram which its phase is the
    // interferometric phase due to the imaging geometry:
    // phase = (4*PI/wavelength)*(rangePixelSpacing)*(rngOffset)
    // complex conjugate of geometryIfgram
    std::valarray<std::complex<float>> geometryIfgramConj(spectrumSize);

    // upsampled interferogram
    std::valarray<std::complex<float>> ifgramUpsampled(_oversampleFactor*ncols*linesPerBlock);

    // full resolution interferogram
    std::valarray<std::complex<float>> ifgram(ncols*linesPerBlock);

    // Looks object and buffers of each output: multi-looked interferogram,
    // power of reference and secondary SLC, coherence for multi-looked and
    // full-res interferogram
    struct LooksBuffers {
        isce3::signal::Looks<float> looksObj;
        std::valarray<std::complex<float>> ifgramMultiLooked;
        std::valarray<float> refPowerLooked, secPowerLooked, coherence;
    };
    std::vector<LooksBuffers> looksBuffers(outputs.size());
    for (size_t k = 0; k < outputs.size(); ++k) {
        const auto& out = outputs[k];
        auto& b = looksBuffers[k];
        if (out.rangeLooks > 1 || out.azimuthLooks > 1) {
            // instantiate Looks used for multi-looking the interferogram
            const size_t linesPerBlockMLooked = linesPerBlock / out.azimuthLooks;
            const size_t ncolsMultiLooked = ncols / out.rangeLooks;
            b.looksObj.nrows(linesPerBlock);
            b.looksObj.ncols(ncols);
            b.looksObj.rowsLooks(out.azimuthLooks);
            b.looksObj.colsLooks(out.rangeLooks);
            b.looksObj.nrowsLooked(linesPerBlockMLooked);
            b.looksObj.ncolsLooked(ncolsMultiLooked);

            // resize following valarrays from empty
            const auto mlookSize = ncolsMultiLooked*linesPerBlockMLooked;
            b.ifgramMultiLooked.resize(mlookSize);
            b.coherence.resize(mlookSize);
            b.refPowerLooked.resize(mlookSize);
            b.secPowerLooked.resize(mlookSize);
        } else {
            b.coherence.resize(ncols*linesPerBlock);
        }
    }

    // storage for spectrum of the block of data in reference SLC
    std::valarray<std::complex<float>> refSpectrum;

    // storage for spectrum of the block of data in secondary SLC
    std::valarray<std::complex<float>> secSpectrum;

    // upsampled spectrum of the block of reference SLC
    std::valarray<std::complex<float>> refSpectrumUpsampled;

    // upsampled spectrum of the block of secondary SLC
    std::valarray<std::complex<float>> secSpectrumUpsampled;

    // upsampled block of reference SLC
    std::valarray<std::complex<float>> refSlcUpsampled;

    // upsampled block of secondary SLC
    std::valarray<std::complex<float>> secSlcUpsampled;

    // only resize valarrays and init FFT when oversampling
    if (_oversampleFactor > 1) {
        refSpectrum.resize(spectrumSize);
        secSpectrum.resize(spectrumSize);

        refSpectrumUpsampled.resize(spectrumUpsampleSize);
        secSpectrumUpsampled.resize(spectrumUpsampleSize);
        refSlcUpsampled.resize(spectrumUpsampleSize);
        secSlcUpsampled.resize(spectrumUpsampleSize);

        // make forward and inverse fft plans for the reference SLC
        refSignal.forwardRangeFFT(refSlc, refSpectrum, fft_size, linesPerBlock);
        refSignal.inverseRangeFFT(refSpectrumUpsampled, refSlcUpsampled,
                fft_size*_oversampleFactor, linesPerBlock);

        // make forward and inverse fft plans for the secondary SLC
        secSignal.forwardRangeFFT(secSlc, secSpectrum, fft_size, linesPerBlock);
        secSignal.inverseRangeFFT(secSpectrumUpsampled, secSlcUpsampled,
                fft_size*_oversampleFactor, linesPerBlock);
    }

    // looking down the upsampled interferogram may shift the samples by a fraction of a pixel
    // depending on the oversample factor. predicting the impact of the shift in frequency domain
    // which is a linear phase allows to account for it during the upsampling process
    std::valarray<std::complex<float>> shiftImpact(spectrumUpsampleSize);
    lookdownShiftImpact(_oversampleFactor,  fft_size,
                        linesPerBlock, shiftImpact);

    // loop over all blocks
    std::cout << "nblocks : " << nblocks << std::endl;

    // raster I/O of the reading thread and of this one, one at a time: the
    // inputs and outputs may be HDF5, whose library is not thread-safe
    std::mutex ioMutex;

    // get a block of reference and secondary SLC data and a block of range
    // offsets into the next-block storage. This zero-pads SLCs in range.
    auto readBlock = [&](size_t block) {
        std::lock_guard<std::mutex> io(ioMutex);
        const auto rowStart = block * linesPerBlock;
        const auto blockRowsData = std::min(nrows - rowStart, linesPerBlock);
        nextRefSlc = 0;
        nextSecSlc = 0;
        // whole blocks (a chunked source, e.g. an HDF5 RSLC, then decodes
        // each chunk once instead of once per line)
        std::valarray<std::complex<float>> data(ncols * blockRowsData);
        refSlcRaster.getBlock(data, 0, rowStart, ncols, blockRowsData);
        for (size_t line = 0; line < blockRowsData; ++line)
            nextRefSlc[std::slice(line*fft_size, ncols, 1)] = data[std::slice(line*ncols, ncols, 1)];
        secSlcRaster.getBlock(data, 0, rowStart, ncols, blockRowsData);
        for (size_t line = 0; line < blockRowsData; ++line)
            nextSecSlc[std::slice(line*fft_size, ncols, 1)] = data[std::slice(line*ncols, ncols, 1)];
        if (flatten) {
            std::valarray<double> offsets(ncols * blockRowsData);
            rngOffsetRaster->getBlock(offsets, 0, rowStart, ncols, blockRowsData);
            nextRngOffset[std::slice(0, ncols * blockRowsData, 1)] =
                offsets + _offsetStartingRangeShift / _rangePixelSpacing;
        }
    };
    auto nextBlock = std::async(std::launch::async, readBlock, 0);

    for (size_t block = 0; block < nblocks; ++block) {
        std::cout << "block: " << block << std::endl;
        // start row for this block
        const auto rowStart = block * linesPerBlock;

        //number of lines of data in this block. blockRowsData<= linesPerBlock
        //Note that linesPerBlock is fixed number of lines
        //blockRowsData might be less than or equal to linesPerBlock.
        //e.g. if nrows = 512, and linesPerBlock = 100, then
        //blockRowsData for last block will be 12
        const auto blockRowsData = std::min(nrows - rowStart, linesPerBlock);

        // take the block read in the background and start reading the next
        // one (the input rasters are only used by the reading thread)
        nextBlock.get();
        std::swap(refSlc, nextRefSlc);
        std::swap(secSlc, nextSecSlc);
        std::swap(rngOffset, nextRngOffset);
        if (block + 1 < nblocks)
            nextBlock = std::async(std::launch::async, readBlock, block + 1);

        // fill the valarray with zero
        ifgramUpsampled = 0;
        ifgram = 0;

        // upsample the reference and secondary SLCs
        if (_oversampleFactor == 1) {
            refSlcUpsampled = refSlc;
            secSlcUpsampled = secSlc;
        } else {
            refSignal.upsample(refSlc, refSlcUpsampled, linesPerBlock, fft_size,
                               _oversampleFactor, shiftImpact);
            secSignal.upsample(secSlc, secSlcUpsampled, linesPerBlock, fft_size,
                               _oversampleFactor, shiftImpact);
        }

        // Compute oversampled interferogram data
        #pragma omp parallel for
        for (size_t line = 0; line < blockRowsData; line++) {
            for (size_t col = 0; col < _oversampleFactor*ncols; col++) {
                ifgramUpsampled[line*(_oversampleFactor*ncols) + col] =
                        refSlcUpsampled[line*(_oversampleFactor*fft_size) + col]*
                        std::conj(secSlcUpsampled[line*(_oversampleFactor*fft_size) + col]);
            }
        }

        if (flatten) {
            #pragma omp parallel for
            for (size_t line = 0; line < blockRowsData; ++line) {
                for (size_t col = 0; col < ncols; ++col) {
                    double phase = 4.0*M_PI*_rangePixelSpacing*rngOffset[line*ncols+col]/_wavelength;
                    geometryIfgramConj[line*fft_size + col] = std::complex<float> (std::cos(phase),
                                                                            -1.0*std::sin(phase));

                }
            }
        }

        // Reclaim the extra oversample looks across
        float ov = _oversampleFactor;
        #pragma omp parallel for
        for (size_t line = 0; line < blockRowsData; line++) {
            for (size_t col = 0; col < ncols; col++) {
                std::complex<float> sum = 0;
                for (size_t j=0; j< _oversampleFactor; j++)
                    sum += ifgramUpsampled[line*(ncols*_oversampleFactor) + j + col*_oversampleFactor];
                ifgram[line*ncols + col] = sum/ov;

                if (flatten)
                    ifgram[line*ncols + col] *= geometryIfgramConj[line*fft_size + col];
            }
        }

        // Take looks down (summing columns), for each output
        for (size_t k = 0; k < outputs.size(); ++k) {
            const int rangeLooks = outputs[k].rangeLooks;
            const int azimuthLooks = outputs[k].azimuthLooks;
            auto& ifgRaster = *outputs[k].ifg;
            auto& coherenceRaster = *outputs[k].coherence;
            auto& looksObj = looksBuffers[k].looksObj;
            auto& ifgramMultiLooked = looksBuffers[k].ifgramMultiLooked;
            auto& refPowerLooked = looksBuffers[k].refPowerLooked;
            auto& secPowerLooked = looksBuffers[k].secPowerLooked;
            auto& coherence = looksBuffers[k].coherence;
            if (rangeLooks > 1 || azimuthLooks > 1) {

                // mulitlook interferogram and set raster
                looksObj.ncols(ncols);
                looksObj.colsLooks(rangeLooks);
                looksObj.multilook(ifgram, ifgramMultiLooked);
                {
                    std::lock_guard<std::mutex> io(ioMutex);
                    ifgRaster.setBlock(ifgramMultiLooked, 0, rowStart/azimuthLooks,
                                ncols/rangeLooks, blockRowsData/azimuthLooks);
                }

                // multilook SLC to power for coherence computation
                // refPowerLooked = average(abs(refSlc)^2)
                if (_oversampleFactor == 1) {
                    looksObj.ncols(fft_size);
                    looksObj.multilook(refSlc, refPowerLooked, 2);
                    looksObj.multilook(secSlc, secPowerLooked, 2);
                } else {
                    // update looksObj so SlcUpsampled can be mulitlooked
                    looksObj.ncols(_oversampleFactor*fft_size);
                    looksObj.colsLooks(_oversampleFactor*rangeLooks);
                    looksObj.multilook(refSlcUpsampled, refPowerLooked, 2);
                    looksObj.multilook(secSlcUpsampled, secPowerLooked, 2);
                }

                // compute coherence
                #pragma omp parallel for
                for (size_t i = 0; i< ifgramMultiLooked.size(); ++i) {
                    coherence[i] = std::abs(ifgramMultiLooked[i])/
                            std::sqrt(refPowerLooked[i]*secPowerLooked[i]);
                }

                // set coherence raster
                {
                    std::lock_guard<std::mutex> io(ioMutex);
                    coherenceRaster.setBlock(coherence, 0, rowStart/azimuthLooks,
                            ncols/rangeLooks, blockRowsData/azimuthLooks);
                }
            } else {
                // fill coherence with ones (no need to compute result)
                coherence = 1.0;

                // set the blocks of interferogram and coherence
                std::lock_guard<std::mutex> io(ioMutex);
                ifgRaster.setBlock(ifgram, 0, rowStart, ncols, blockRowsData);
                coherenceRaster.setBlock(coherence, 0, rowStart, ncols,
                                         blockRowsData);
            }
        }
    }
}
