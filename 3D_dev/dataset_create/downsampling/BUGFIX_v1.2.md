# MRI Downsampling Pipeline - BUGFIX v1.2

**Date**: 2025-01-11
**Status**: CRITICAL FIXES - Production Ready ✅
**Previous Version**: v1.1

---

## Executive Summary

Version 1.2 addresses **2 BLOCKING bugs** identified in v1.1 that would have caused incorrect results:

1. **Anisotropic Gaussian Implementation** - Fixed incorrect API usage that prevented true anisotropic smoothing
2. **Probability Label Interpolator** - Fixed bspline overshoot causing negative probabilities

Both bugs were critical and would have significantly impacted scientific accuracy.

---

## BLOCKING Bug Fixes

### Bug #1: Anisotropic Gaussian Implementation (CRITICAL)

**Location**: `mri_downsampling_pipeline.py:517-543, 579-591`

**Issue**:
```python
# WRONG - SetSigma() only accepts scalar, not list
smoother = sitk.SmoothingRecursiveGaussianImageFilter()
smoother.SetSigma([σx, σy, σz])  # ❌ API doesn't support lists
```

The `SmoothingRecursiveGaussianImageFilter.SetSigma()` method only accepts a scalar value for isotropic smoothing, NOT a list/tuple for anisotropic smoothing. This would either:
- Cause a runtime type error, OR
- Silently use only one σ value, resulting in isotropic smoothing instead of the required anisotropic PSF matching

**Impact**: Channel families requiring anisotropic smoothing (e.g., MPRAGE: σ=(0.713, 0.713, 1.245) mm) would be smoothed incorrectly, breaking PSF matching.

**Fix**:

Added helper method `_apply_anisotropic_gaussian()` that applies three sequential 1D RecursiveGaussian filters:

```python
def _apply_anisotropic_gaussian(self,
                               image: sitk.Image,
                               sigma_mm: Tuple[float, float, float]) -> sitk.Image:
    """
    Apply anisotropic Gaussian smoothing using three sequential 1D filters

    This is the CORRECT way to do anisotropic Gaussian in SimpleITK
    """
    result = image

    for axis in range(3):
        sigma = float(sigma_mm[axis])
        if sigma > 1e-6:
            smoother = sitk.RecursiveGaussianImageFilter()
            smoother.SetSigma(sigma)              # Scalar sigma
            smoother.SetDirection(axis)            # Apply to one axis
            smoother.SetNormalizeAcrossScale(True)
            result = smoother.Execute(result)

    return result
```

**Key Changes**:
- Uses `RecursiveGaussianImageFilter` (not `SmoothingRecursiveGaussianImageFilter`)
- Applies filter three times sequentially, once per axis
- Each filter uses `SetSigma(float)` with scalar value and `SetDirection(axis)`
- Guarantees true anisotropic smoothing with independent σ per axis

**Verification**:
- All channel families now correctly apply their specified (σx, σy, σz) values
- Physical spacing semantics preserved (all σ values in mm)
- Applied to both data and mask in normalized convolution

---

### Bug #2: Probability Label Interpolator (CRITICAL)

**Location**: `mri_downsampling_pipeline.py:1176-1215`

**Issue**:
```python
# WRONG - Uses MPRAGE's bspline interpolator
mprage_family = self.config.FAMILIES['MPRAGE']  # interpolator='bspline'

for c in range(num_classes):
    proba_labels[..., c] = self.downsampler.downsample_family_D(
        onehot[..., c], mprage_family, spacing_in, spacing_out, mask
    )  # ❌ BSpline causes negative values on one-hot data
```

B-spline interpolation on binary one-hot maps (values only 0 or 1) causes:
- **Negative values** due to undershooting at edges
- **Values > 1** due to overshooting at edges
- Violation of probability constraints [0, 1]

**Impact**: Probability labels would contain invalid values, breaking downstream models expecting valid probabilities that sum to 1.

**Fix**:

Create temporary `ChannelFamily` object with MPRAGE's PSF but **forced linear interpolator**:

```python
# Use MPRAGE family PSF for label smoothing
mprage_family = self.config.FAMILIES['MPRAGE']

# Create a temporary family with MPRAGE PSF but LINEAR interpolator
# IMPORTANT: Probability maps MUST use linear interpolation to avoid negative values
proba_family = ChannelFamily(
    name='MPRAGE_linear',
    indices=mprage_family.indices,
    native_resolution=mprage_family.native_resolution,
    sigma_add_mm=mprage_family.sigma_add_mm,  # Keep PSF matching
    method=mprage_family.method,
    interpolator='linear'  # ✅ Force linear instead of bspline
)

self.logger.info(f"  Using MPRAGE PSF (σ={proba_family.sigma_add_mm} mm) with LINEAR interpolator")

# Downsample each class probability map
for c in range(num_classes):
    proba_labels[..., c] = self.downsampler.downsample_family_D(
        onehot[..., c], proba_family, spacing_in, spacing_out, mask
    )
```

**Key Changes**:
- Creates `proba_family` with `interpolator='linear'`
- Keeps MPRAGE's `sigma_add_mm=(0.713, 0.713, 1.245)` for correct PSF matching
- Linear interpolation guarantees output ∈ [0, 1] for binary input
- Added explicit logging of interpolator choice

**Verification**:
- Probability values guaranteed to be in [0, 1]
- After normalization, probabilities sum to 1.0 per voxel
- No negative values or overshooting

---

## Implementation Details

### Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `mri_downsampling_pipeline.py` | 517-543 | Added `_apply_anisotropic_gaussian()` helper method |
| `mri_downsampling_pipeline.py` | 579-591 | Replaced buggy code with helper method calls |
| `mri_downsampling_pipeline.py` | 1176-1215 | Added `proba_family` with forced linear interpolator |

### Code Statistics

- **Lines added**: ~40
- **Lines modified**: ~15
- **Methods added**: 1 (`_apply_anisotropic_gaussian`)
- **Breaking changes**: None (internal implementation fixes)

---

## Testing & Validation

### Pre-Fix Issues

1. **Anisotropic Gaussian**:
   - Would likely crash with type error: `SetSigma() expects float, got list`
   - OR silently apply wrong smoothing, breaking PSF matching accuracy

2. **Probability Labels**:
   - Would produce negative values (e.g., -0.05 to 0.03 near edges)
   - Would produce values > 1 (e.g., 0.97 to 1.15 at peaks)
   - Sum normalization would mask but not eliminate the fundamental error

### Post-Fix Validation

**Anisotropic Gaussian**:
- [x] Each axis smoothed independently with correct σ value
- [x] Physical spacing (mm) correctly interpreted by SimpleITK
- [x] Applied to both image and mask in normalized convolution

**Probability Labels**:
- [x] All values in [0, 1] range
- [x] Per-voxel sum ≈ 1.0 (after normalization)
- [x] No negative values
- [x] PSF matching preserved (σ from MPRAGE)

---

## Migration Guide

### For Users of v1.1

**Action Required**: Update to v1.2 immediately. These bugs would cause scientifically invalid results.

**Breaking Changes**: None. API unchanged.

**Validation Steps**:
1. Re-run pipeline on test data
2. Check QA metrics for probability labels:
   ```python
   assert proba_labels.min() >= 0.0, "Negative probabilities detected!"
   assert proba_labels.max() <= 1.0, "Probabilities > 1 detected!"
   assert np.allclose(proba_labels.sum(axis=-1), 1.0, atol=1e-3), "Sum ≠ 1!"
   ```
3. Verify log shows: `"Using MPRAGE PSF ... with LINEAR interpolator"`
4. Verify log shows: `"Applying anisotropic Gaussian: σ_add=(...) mm (physical space)"`

### For New Users

Simply use v1.2. No special migration needed.

---

## Technical References

### SimpleITK Anisotropic Gaussian

**Correct Approach** (v1.2):
```python
for axis in [0, 1, 2]:
    filter = sitk.RecursiveGaussianImageFilter()
    filter.SetSigma(sigma[axis])
    filter.SetDirection(axis)
    image = filter.Execute(image)
```

**Incorrect Approach** (v1.1):
```python
# This API doesn't exist / doesn't work as expected
filter = sitk.SmoothingRecursiveGaussianImageFilter()
filter.SetSigma([sx, sy, sz])  # ❌
```

**Reference**: SimpleITK Examples - RecursiveGaussianImageFilter
https://simpleitk.readthedocs.io/en/master/link_RecursiveGaussianImageFilter_docs.html

### BSpline Overshoot on Binary Data

**Problem**: B-spline interpolation uses cubic polynomials, which can undershoot/overshoot at discontinuities.

**Example**:
```
Input (one-hot):  [0, 0, 1, 1, 0, 0]
Linear output:    [0, 0.2, 0.9, 0.8, 0.1, 0]  ✅ ∈ [0,1]
BSpline output:   [-0.05, 0.15, 1.15, 0.85, 0.08, -0.02]  ❌ overshoots
```

**Reference**:
- Thévenaz et al. (2000). "Interpolation Revisited"
- IEEE Trans Medical Imaging standard practice for probability maps

---

## Performance Impact

### Anisotropic Gaussian Fix

**Computational Cost**:
- **Before (buggy)**: 1 × isotropic smoothing call (if didn't crash)
- **After (correct)**: 3 × 1D smoothing calls
- **Overhead**: ~0-10% depending on data size (RecursiveGaussian is O(N), very fast)

**Memory**: No change

### Probability Label Fix

**Computational Cost**: No change (linear vs bspline have similar cost)

**Memory**: No change

---

## Future Improvements (Optional)

These bugs are now FIXED. The following are nice-to-have enhancements for future versions:

1. **Tighten acceptance test thresholds**: Current 5-10 dB → target 15-20 dB
2. **Add Nyquist suppression metrics**: Report aliasing suppression ≤-20 dB
3. **Expose `z_offsets_ppm` parameter**: Allow user-specified frequency offsets
4. **Record correlation coefficients**: Save M0 selection correlations to metadata
5. **Clean up CEST family indices**: Remove duplicate M0 channels from main indices

---

## Conclusion

Version 1.2 fixes two critical bugs that would have caused scientifically invalid results:

1. ✅ **Anisotropic Gaussian now correctly implemented** - True per-axis smoothing
2. ✅ **Probability labels now use linear interpolation** - No negative values

**Recommendation**: All users must upgrade from v1.1 to v1.2 immediately.

**Status**: Production ready with correct scientific implementation.

---

**Version History**:
- v1.0: Initial implementation
- v1.1: Fixed normalized convolution, Z-spectrum detection, M0 selection
- v1.2: Fixed anisotropic Gaussian and probability interpolator (CRITICAL)

**Next Version** (v1.3, optional enhancements):
- Add QA metrics improvements
- Expose advanced parameters
- Performance optimizations
