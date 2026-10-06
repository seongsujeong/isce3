#!/usr/bin/env python3

import isce3
import journal
import numpy as np
import pytest

from nisar.workflows.offsets_product import (clip_gross_offsets,
                                             get_start_pixels)


def make_cfg(windows, start=None):
    '''Minimal offsets_product cfg with square layer windows'''
    cfg = {'margin': 50, 'gross_offset_range': 0, 'gross_offset_azimuth': 0,
           'start_pixel_range': start, 'start_pixel_azimuth': start}
    for i, win in enumerate(windows):
        cfg[f'layer{i + 1}'] = {'window_range': win, 'window_azimuth': win,
                                'half_search_range': 20,
                                'half_search_azimuth': 20}
    return cfg


@pytest.mark.parametrize("windows", [(32, 64, 128), (64, 96, 196),
                                     (33, 64, 127)])
@pytest.mark.parametrize("start", [None, 314])
def test_layer_windows_centered_on_grid(windows, start):
    cfg = make_cfg(windows, start)

    # Common start pixel (margin + half search if not set), and grid center
    # as in helpers.get_offset_radar_grid
    start0 = 50 + 20 if start is None else start
    center = start0 + min(windows) // 2

    for win in windows:
        az_start, rg_start = get_start_pixels(cfg, win, win)
        assert rg_start + win // 2 == center
        assert az_start + win // 2 == center

    # Smallest window keeps the common start pixel
    assert get_start_pixels(cfg, min(windows), min(windows)) == (start0, start0)


@pytest.mark.parametrize("window_azimuth, window_range",
                         [(None, 64), (64, None), (None, None)])
def test_missing_window_size(window_azimuth, window_range):
    with pytest.raises(journal.ApplicationError):
        get_start_pixels(make_cfg((32, 64, 128)), window_azimuth,
                         window_range)


def test_clip_gross_offsets():
    '''Secondary search windows must stay inside the secondary image'''
    amp = isce3.matchtemplate.PyCPUAmpcor()
    amp.windowSizeHeight, amp.windowSizeWidth = 32, 64
    amp.halfSearchRangeDown, amp.halfSearchRangeAcross = 16, 8
    amp.skipSampleDown, amp.skipSampleAcross = 10, 20
    amp.referenceStartPixelDownStatic = 20
    amp.referenceStartPixelAcrossStatic = 10
    amp.numberWindowDown, amp.numberWindowAcross = 5, 4
    amp.secondaryImageHeight, amp.secondaryImageWidth = 120, 200

    n = 5 * 4
    for big in (-1000, 0, 3, 1000):
        az, rg = clip_gross_offsets(amp, np.full(n, big, np.int32),
                                    np.full(n, big, np.int32))
        ref_az = (20 + 10 * np.arange(5))[:, None] + np.zeros((5, 4), int)
        ref_rg = (10 + 20 * np.arange(4))[None, :] + np.zeros((5, 4), int)
        sec_az = ref_az.ravel() + az - 16
        sec_rg = ref_rg.ravel() + rg - 8
        assert sec_az.min() >= 0 and sec_rg.min() >= 0
        assert (sec_az + 32 + 2 * 16).max() < 120
        assert (sec_rg + 64 + 2 * 8).max() < 200
    # offsets that already fit are unchanged
    az, rg = clip_gross_offsets(amp, np.full(n, 3, np.int32),
                                np.full(n, -2, np.int32))
    assert (az == 3).all() and (rg == -2).all()
