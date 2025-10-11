"""
MRI Multi-modal Downsampling Pipeline to CEST Resolution
=========================================================

Scientific downsampling of 4D multi-modal MRI data from high-resolution
(~0.65mm isotropic) to CEST acquisition resolution (1.8×1.8×3.0 mm³).

Author: Generated for KAN-Brain project
Date: 2025-01-11
Version: 1.0.0

Key Features:
- Channel-family-specific PSF matching (anisotropic Gaussian in mm space)
- Method D: Anti-aliasing Gaussian + Linear resampling for intensity data
- Method C: Re-computation on low-resolution for ratio/parametric maps
- Automatic CEST slab localization with robust fallback
- Comprehensive QA metrics and detailed logging
- 100% reproducible with fixed random seeds
"""

import numpy as np
import SimpleITK as sitk
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from scipy import stats
from scipy.ndimage import label as nd_label
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================================
# Configuration & Channel Definitions
# ==============================================================================

@dataclass
class ChannelFamily:
    """Channel family configuration with PSF parameters"""
    name: str
    indices: List[int]  # 0-based Python indices
    native_resolution: Tuple[float, float, float]  # mm: (X, Y, Z)
    sigma_add_mm: Tuple[float, float, float]  # mm: additional Gaussian sigma
    method: str  # 'D' (direct downsample) or 'C' (recompute)
    interpolator: str  # 'linear', 'bspline', 'nearest'


class ChannelConfig:
    """Channel configuration based on paper specification (1-based indices)"""

    # Target CEST resolution
    TARGET_SPACING_MM = (1.8, 1.8, 3.0)  # (X, Y, Z) in mm

    # Input high-resolution spacing (from FOV)
    INPUT_SPACING_MM = (218/336, 166/256, 250/384)  # ≈ (0.649, 0.648, 0.651)

    # Channel families with PSF parameters
    FAMILIES = {
        'CEST': ChannelFamily(
            name='CEST',
            # Z-spectra only (M0 channels handled separately in _process_cest_family)
            # Low B1: 231-284 (1-based) = 230-283 (0-based) = 54 points
            # High B1: 287-340 (1-based) = 286-339 (0-based) = 54 points
            indices=list(range(230, 284)) + list(range(286, 340)),
            native_resolution=(1.8, 1.8, 3.0),
            sigma_add_mm=(0.0, 0.0, 0.0),
            method='C',  # Z-spectra need special handling
            interpolator='linear'
        ),
        'QTI_params': ChannelFamily(
            name='QTI_params',
            indices=list(range(0, 15)),  # 1-15 in 1-based
            native_resolution=(1.5, 1.5, 3.0),
            sigma_add_mm=(0.423, 0.423, 0.0),
            method='C',  # Refit from low-res DWI
            interpolator='bspline'
        ),
        'DWI': ChannelFamily(
            name='DWI',
            indices=list(range(15, 225)),  # 16-225 in 1-based (b_lin, b_plan, b_spher)
            native_resolution=(1.5, 1.5, 3.0),
            sigma_add_mm=(0.423, 0.423, 0.0),
            method='D',
            interpolator='linear'
        ),
        'CEST_params': ChannelFamily(
            name='CEST_params',
            indices=list(range(225, 229)),  # 226-229 in 1-based
            native_resolution=(1.8, 1.8, 3.0),
            sigma_add_mm=(0.0, 0.0, 0.0),
            method='C',  # Refit from low-res Z-spectra
            interpolator='bspline'
        ),
        'MPRAGE': ChannelFamily(
            name='MPRAGE',
            indices=[341],  # 342 in 1-based
            native_resolution=(0.65, 0.65, 0.65),
            sigma_add_mm=(0.713, 0.713, 1.245),
            method='D',
            interpolator='bspline'
        ),
        'GRE': ChannelFamily(
            name='GRE',
            indices=list(range(342, 347)),  # 343-347 in 1-based (QSM_TE)
            native_resolution=(0.6, 0.6, 0.6),
            sigma_add_mm=(0.721, 0.721, 1.249),
            method='D',
            interpolator='linear'
        ),
        'TE_avg': ChannelFamily(
            name='TE_avg',
            indices=[347],  # 348 in 1-based
            native_resolution=(0.65, 0.65, 0.65),
            sigma_add_mm=(0.713, 0.713, 1.245),
            method='D',
            interpolator='linear'
        ),
        'SMWI': ChannelFamily(
            name='SMWI',
            indices=[348, 349],  # 349-350 in 1-based
            native_resolution=(0.6, 0.6, 0.6),
            sigma_add_mm=(0.721, 0.721, 1.249),
            method='C',  # Recompute from low-res multi-echo
            interpolator='bspline'
        ),
        'QSM': ChannelFamily(
            name='QSM',
            indices=[350],  # 351 in 1-based
            native_resolution=(0.6, 0.6, 0.6),
            sigma_add_mm=(0.721, 0.721, 1.249),
            method='C',  # Recompute from low-res phase
            interpolator='bspline'
        ),
    }

    # Z-spectrum and M0 channel mappings (0-based)
    Z_SPECTRUM_LOW_B1 = list(range(230, 284))  # 231-284 in 1-based (54 points)
    Z_SPECTRUM_HIGH_B1 = list(range(286, 340))  # 287-340 in 1-based (54 points)
    M0_LOW_B1 = 229  # 230 in 1-based
    M0_HIGH_B1_PRIMARY = 284  # 285 in 1-based
    M0_HIGH_B1_SECONDARY = 285  # 286 in 1-based
    M0_FALLBACK = 340  # 341 in 1-based

    # Z-spectrum frequency offsets (ppm) - typical CEST Z-spectrum
    # Default: symmetric around 0 ppm, ranging from -5 to +5 ppm
    # 54 points with ~0.19 ppm spacing
    Z_SPECTRUM_OFFSETS_PPM = np.linspace(-5.0, 5.0, 54)

    # Far-off-resonance threshold for normalization detection
    FAR_OFFRESONANCE_THRESHOLD_PPM = 4.0  # |Δω| > 4 ppm considered far-off-resonance

    @classmethod
    def get_family_for_channel(cls, channel_idx: int) -> Optional[ChannelFamily]:
        """Get the channel family for a given 0-based channel index"""
        for family in cls.FAMILIES.values():
            if channel_idx in family.indices:
                return family
        return None


# ==============================================================================
# Utility Functions
# ==============================================================================

def setup_logging(output_dir: Path, log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'downsampling_{timestamp}.log'

    logger = logging.getLogger('MRI_Downsampling')

    # Clear existing handlers to prevent duplication on multiple initializations
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(getattr(logging, log_level.upper()))

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def numpy_to_sitk(array: np.ndarray, spacing: Tuple[float, float, float],
                  origin: Tuple[float, float, float] = (0., 0., 0.)) -> sitk.Image:
    """Convert NumPy array to SimpleITK image with proper metadata"""
    # SimpleITK expects (X, Y, Z) order
    if array.ndim == 3:
        image = sitk.GetImageFromArray(array.transpose(2, 1, 0))  # (X,Y,Z) -> (Z,Y,X) for SITK
    else:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # Identity direction

    return image


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to NumPy array in (X, Y, Z) order"""
    array = sitk.GetArrayFromImage(image)  # Returns (Z, Y, X)
    return array.transpose(2, 1, 0)  # Convert to (X, Y, Z)


def fix_random_seeds(seed: int = 42):
    """Fix random seeds for reproducibility"""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ==============================================================================
# Step 0: Axis Reordering & Assertions
# ==============================================================================

class AxisReorderer:
    """Handle axis reordering from (Z,X,Y,C) to (X,Y,Z,C)"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def reorder_axes(self,
                     data: np.ndarray,
                     region_mask: np.ndarray,
                     region_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Reorder axes from (Z,X,Y,C) to (X,Y,Z,C)

        Args:
            data: (384, 336, 256, 351) - (Z, X, Y, C)
            region_mask: (384, 336, 256) - (Z, X, Y)
            region_labels: (384, 336, 256) - (Z, X, Y)

        Returns:
            Dict with reordered arrays in (X, Y, Z, C) format
        """
        self.logger.info("=" * 80)
        self.logger.info("Step 0: Axis Reordering & Assertions")
        self.logger.info("=" * 80)

        # Input validation
        expected_shape = (384, 336, 256)
        assert data.shape[:3] == expected_shape, \
            f"Expected data shape {expected_shape}+(351,), got {data.shape}"
        assert region_mask.shape == expected_shape, \
            f"Expected mask shape {expected_shape}, got {region_mask.shape}"
        assert region_labels.shape == expected_shape, \
            f"Expected labels shape {expected_shape}, got {region_labels.shape}"

        self.logger.info(f"Input shapes validated:")
        self.logger.info(f"  data: {data.shape}")
        self.logger.info(f"  region_mask: {region_mask.shape}")
        self.logger.info(f"  region_labels: {region_labels.shape}")

        # Reorder: (Z, X, Y, C) -> (X, Y, Z, C)
        # Permutation: (1, 2, 0, 3) for 4D, (1, 2, 0) for 3D
        data_reordered = np.transpose(data, (1, 2, 0, 3))
        mask_reordered = np.transpose(region_mask, (1, 2, 0))
        labels_reordered = np.transpose(region_labels, (1, 2, 0))

        # Assertion 1: Check reordered shape
        expected_reordered = (336, 256, 384)
        assert data_reordered.shape[:3] == expected_reordered, \
            f"After reordering, expected {expected_reordered}+(351,), got {data_reordered.shape}"

        self.logger.info(f"✓ Assertion 1 passed: Reordered shape = {data_reordered.shape}")

        # Store for later slab thickness check (done in SlabLocalizer)
        result = {
            'data': data_reordered,
            'region_mask': mask_reordered,
            'region_labels': labels_reordered,
            'input_spacing': ChannelConfig.INPUT_SPACING_MM,
            'physical_size_mm': tuple(
                s * sp for s, sp in zip(expected_reordered, ChannelConfig.INPUT_SPACING_MM)
            )
        }

        self.logger.info(f"Physical size (X,Y,Z): {result['physical_size_mm']} mm")
        self.logger.info(f"Input spacing: {result['input_spacing']} mm")

        return result


# ==============================================================================
# Step 1: CEST Slab Localization & Cropping
# ==============================================================================

class SlabLocalizer:
    """Automatic CEST slab localization with robust fallback"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.config = ChannelConfig()

    def localize_and_crop(self,
                         data: np.ndarray,
                         region_mask: np.ndarray,
                         region_labels: np.ndarray,
                         spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Automatically locate and crop to CEST coverage slab

        Args:
            data: (X, Y, Z, 351) in mm-space order
            region_mask: (X, Y, Z)
            region_labels: (X, Y, Z)
            spacing: (X, Y, Z) spacing in mm

        Returns:
            Dict with cropped data and metadata
        """
        self.logger.info("=" * 80)
        self.logger.info("Step 1: CEST Slab Localization & Cropping")
        self.logger.info("=" * 80)

        X, Y, Z, C = data.shape

        # Step 1: Compute Z-spectrum intensity maps
        self.logger.info("Computing Z-spectrum intensity maps...")
        z_low = data[:, :, :, self.config.Z_SPECTRUM_LOW_B1]  # (X,Y,Z,54)
        z_high = data[:, :, :, self.config.Z_SPECTRUM_HIGH_B1]  # (X,Y,Z,54)

        # Sum across frequency offsets
        zsum_low = np.sum(z_low, axis=-1)  # (X,Y,Z)
        zsum_high = np.sum(z_high, axis=-1)  # (X,Y,Z)
        zsum = np.maximum(zsum_low, zsum_high)

        # Step 2: Get M0 maps - use ALL M0 candidates for robustness
        m0_low = data[:, :, :, self.config.M0_LOW_B1]
        m0_high_primary = data[:, :, :, self.config.M0_HIGH_B1_PRIMARY]
        m0_high_secondary = data[:, :, :, self.config.M0_HIGH_B1_SECONDARY]
        m0_fallback = data[:, :, :, self.config.M0_FALLBACK]

        # Take maximum across all M0 candidates for most robust slab detection
        m0 = np.maximum.reduce([m0_low, m0_high_primary, m0_high_secondary, m0_fallback])

        # Step 3: Adaptive thresholding
        self.logger.info("Computing adaptive thresholds...")

        # Use brain mask to compute thresholds only within brain
        brain_voxels_zsum = zsum[region_mask > 0]
        brain_voxels_m0 = m0[region_mask > 0]

        # Use percentile-based thresholds
        tau1 = np.percentile(brain_voxels_zsum, 90) * 0.3  # Conservative
        tau2 = np.percentile(brain_voxels_m0, 90) * 0.3

        self.logger.info(f"Thresholds: τ1={tau1:.2f}, τ2={tau2:.2f}")

        # Candidate mask
        candidate_mask = (zsum > tau1) & (m0 > tau2) & (region_mask > 0)

        # Step 4: 3D connected component analysis
        self.logger.info("Performing 3D connected component analysis...")
        labeled_array, num_features = nd_label(candidate_mask)

        if num_features == 0:
            self.logger.warning("No connected components found! Using fallback...")
            return self._fallback_cropping(data, region_mask, region_labels, spacing)

        # Find largest connected component
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # Ignore background
        largest_component = np.argmax(component_sizes)

        # Get bounding box of largest component
        slab_mask = (labeled_array == largest_component)
        bbox = self._get_bounding_box(slab_mask)

        self.logger.info(f"Bounding box (X,Y,Z): {bbox}")

        # Step 5: Validate Z-axis thickness
        z_thickness_voxels = bbox[2][1] - bbox[2][0]
        z_thickness_mm = z_thickness_voxels * spacing[2]

        self.logger.info(f"Z-axis thickness: {z_thickness_mm:.2f} mm ({z_thickness_voxels} voxels)")

        # Assertion 2: Check slab thickness
        if not (48 <= z_thickness_mm <= 60):
            self.logger.warning(
                f"⚠️  Slab thickness {z_thickness_mm:.2f} mm outside expected range [48, 60] mm!"
            )

            # Try lowering threshold
            self.logger.info("Retrying with lower threshold (p80)...")
            tau1 = np.percentile(brain_voxels_zsum, 80) * 0.3
            tau2 = np.percentile(brain_voxels_m0, 80) * 0.3
            candidate_mask = (zsum > tau1) & (m0 > tau2) & (region_mask > 0)

            labeled_array, num_features = nd_label(candidate_mask)
            if num_features > 0:
                component_sizes = np.bincount(labeled_array.ravel())
                component_sizes[0] = 0
                largest_component = np.argmax(component_sizes)
                slab_mask = (labeled_array == largest_component)
                bbox = self._get_bounding_box(slab_mask)
                z_thickness_mm = (bbox[2][1] - bbox[2][0]) * spacing[2]

                if 48 <= z_thickness_mm <= 60:
                    self.logger.info(f"✓ Retry successful: {z_thickness_mm:.2f} mm")
                else:
                    self.logger.warning("Retry failed, using fallback...")
                    return self._fallback_cropping(data, region_mask, region_labels, spacing)
            else:
                self.logger.warning("Retry failed, using fallback...")
                return self._fallback_cropping(data, region_mask, region_labels, spacing)
        else:
            self.logger.info(f"✓ Assertion 2 passed: Slab thickness = {z_thickness_mm:.2f} mm")

        # Crop data
        x_slice = slice(bbox[0][0], bbox[0][1])
        y_slice = slice(bbox[1][0], bbox[1][1])
        z_slice = slice(bbox[2][0], bbox[2][1])

        cropped_data = data[x_slice, y_slice, z_slice, :]
        cropped_mask = region_mask[x_slice, y_slice, z_slice]
        cropped_labels = region_labels[x_slice, y_slice, z_slice]
        cropped_slab_mask = slab_mask[x_slice, y_slice, z_slice]

        result = {
            'data': cropped_data,
            'region_mask': cropped_mask,
            'region_labels': cropped_labels,
            'slab_mask': cropped_slab_mask,
            'bbox': bbox,
            'slab_thickness_mm': z_thickness_mm,
            'slab_thickness_voxels': z_thickness_voxels,
            'coverage_fallback': False,
            'thresholds': {'tau1': float(tau1), 'tau2': float(tau2)},
            'cropped_shape': cropped_data.shape[:3],
            'physical_size_mm': tuple(
                s * sp for s, sp in zip(cropped_data.shape[:3], spacing)
            )
        }

        self.logger.info(f"Cropped shape: {result['cropped_shape']}")
        self.logger.info(f"Physical size: {result['physical_size_mm']} mm")

        return result

    def _get_bounding_box(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Get 3D bounding box of binary mask"""
        coords = np.argwhere(mask)
        bbox = []
        for axis in range(3):
            bbox.append((coords[:, axis].min(), coords[:, axis].max() + 1))
        return bbox

    def _fallback_cropping(self,
                          data: np.ndarray,
                          region_mask: np.ndarray,
                          region_labels: np.ndarray,
                          spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """Fallback: center crop with ~54mm thickness"""
        self.logger.warning("Using fallback: center crop with 54mm thickness")

        X, Y, Z, C = data.shape
        target_z_mm = 54.0
        target_z_voxels = int(np.round(target_z_mm / spacing[2]))

        # Center crop in Z
        z_start = (Z - target_z_voxels) // 2
        z_end = z_start + target_z_voxels

        # Full X, Y extent
        bbox = [(0, X), (0, Y), (z_start, z_end)]

        cropped_data = data[:, :, z_start:z_end, :]
        cropped_mask = region_mask[:, :, z_start:z_end]
        cropped_labels = region_labels[:, :, z_start:z_end]

        return {
            'data': cropped_data,
            'region_mask': cropped_mask,
            'region_labels': cropped_labels,
            'slab_mask': cropped_mask,  # Use brain mask as slab mask
            'bbox': bbox,
            'slab_thickness_mm': target_z_voxels * spacing[2],
            'slab_thickness_voxels': target_z_voxels,
            'coverage_fallback': True,
            'thresholds': {'tau1': None, 'tau2': None},
            'cropped_shape': cropped_data.shape[:3],
            'physical_size_mm': tuple(
                s * sp for s, sp in zip(cropped_data.shape[:3], spacing)
            )
        }


# ==============================================================================
# Step 2-3: Channel-Family-Specific Downsampling (Method D)
# ==============================================================================

class ChannelDownsampler:
    """Handle channel-family-specific downsampling with PSF matching"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.config = ChannelConfig()

    def _apply_anisotropic_gaussian(self,
                                   image: sitk.Image,
                                   sigma_mm: Tuple[float, float, float]) -> sitk.Image:
        """
        Apply anisotropic Gaussian smoothing using three sequential 1D filters

        Args:
            image: SimpleITK image
            sigma_mm: Gaussian sigma in mm for (X, Y, Z) axes

        Returns:
            Smoothed image
        """
        # Apply 1D RecursiveGaussian filter sequentially per axis
        # This is the CORRECT way to do anisotropic Gaussian in SimpleITK
        result = image

        for axis in range(3):
            sigma = float(sigma_mm[axis])
            if sigma > 1e-6:
                smoother = sitk.RecursiveGaussianImageFilter()
                smoother.SetSigma(sigma)
                smoother.SetDirection(axis)
                smoother.SetNormalizeAcrossScale(True)
                result = smoother.Execute(result)

        return result

    def downsample_family_D(self,
                           data_3d: np.ndarray,
                           family: ChannelFamily,
                           spacing_in: Tuple[float, float, float],
                           spacing_out: Tuple[float, float, float],
                           mask: Optional[np.ndarray] = None,
                           use_normalized_conv: bool = True) -> np.ndarray:
        """
        Method D: Anti-aliasing Gaussian + Linear resampling with optional mask edge correction

        Args:
            data_3d: (X, Y, Z) single channel data
            family: ChannelFamily configuration
            spacing_in: Input spacing in mm
            spacing_out: Target spacing in mm
            mask: Optional mask for edge correction (normalized convolution)
            use_normalized_conv: Apply normalized convolution for edge correction (default: True)

        Returns:
            Downsampled (X', Y', Z') data
        """
        # Apply mask edge correction if requested and mask provided
        if use_normalized_conv and mask is not None:
            # Normalized convolution: out = (G * (img·mask)) / (G * mask + ε)
            data_masked = data_3d * mask.astype(data_3d.dtype)

            # Convert both to SimpleITK
            image = numpy_to_sitk(data_masked, spacing_in)
            mask_image = numpy_to_sitk(mask.astype(np.float32), spacing_in)
        else:
            # Standard path without edge correction
            image = numpy_to_sitk(data_3d, spacing_in)
            mask_image = None

        # Apply anisotropic Gaussian smoothing (PSF matching)
        # Use three sequential 1D RecursiveGaussian filters for correct anisotropic smoothing
        if any(s > 1e-6 for s in family.sigma_add_mm):
            self.logger.debug(
                f"  Applying anisotropic Gaussian: σ_add={family.sigma_add_mm} mm (physical space)"
            )

            # Apply smoothing using helper method
            image = self._apply_anisotropic_gaussian(image, family.sigma_add_mm)

            # Apply same smoothing to mask if using normalized convolution
            if mask_image is not None:
                mask_image = self._apply_anisotropic_gaussian(mask_image, family.sigma_add_mm)

        # Resample to target spacing
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        # Calculate new size
        new_size = [
            int(np.round(original_size[i] * original_spacing[i] / spacing_out[i]))
            for i in range(3)
        ]

        self.logger.debug(
            f"  Resampling: {original_size} @ {original_spacing} -> "
            f"{new_size} @ {spacing_out}"
        )

        # Setup resampler for data
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(spacing_out)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0.0)

        # Set interpolator for data
        if family.interpolator == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif family.interpolator == 'bspline':
            resampler.SetInterpolator(sitk.sitkBSpline)
        elif family.interpolator == 'nearest':
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            raise ValueError(f"Unknown interpolator: {family.interpolator}")

        # Execute resampling for data
        resampled = resampler.Execute(image)

        # If using normalized convolution, also resample mask
        # IMPORTANT: Always use LINEAR for mask to avoid BSpline overshoot
        if mask_image is not None:
            # Create separate resampler for mask with LINEAR interpolation
            mask_resampler = sitk.ResampleImageFilter()
            mask_resampler.SetSize(new_size)
            mask_resampler.SetOutputSpacing(spacing_out)
            mask_resampler.SetOutputOrigin(mask_image.GetOrigin())
            mask_resampler.SetOutputDirection(mask_image.GetDirection())
            mask_resampler.SetTransform(sitk.Transform())
            mask_resampler.SetInterpolator(sitk.sitkLinear)  # Always LINEAR for mask
            mask_resampler.SetDefaultPixelValue(0.0)

            resampled_mask = mask_resampler.Execute(mask_image)

            # Normalized convolution: divide by smoothed mask
            # Convert to NumPy for easier manipulation
            data_resampled = sitk_to_numpy(resampled)
            mask_resampled = sitk_to_numpy(resampled_mask)

            # Perform division with epsilon for numerical stability
            epsilon = 1e-10
            result = data_resampled / (mask_resampled + epsilon)

            # Clip to zero where mask is very small (background)
            result = np.where(mask_resampled > 0.01, result, 0.0)
        else:
            # Convert back to NumPy
            result = sitk_to_numpy(resampled)

        return result


# ==============================================================================
# Step 4: Z-spectrum Method C (Re-normalization)
# ==============================================================================

class ZSpectrumProcessor:
    """Handle Z-spectrum specific processing (Method C)"""

    def __init__(self, logger: logging.Logger, downsampler: ChannelDownsampler):
        self.logger = logger
        self.config = ChannelConfig()
        self.downsampler = downsampler

    def select_best_m0(self,
                      z_spectrum: np.ndarray,
                      m0_candidates: Dict[str, np.ndarray],
                      mask: np.ndarray) -> Tuple[str, np.ndarray, Dict[str, float]]:
        """
        Select best M0 map based on correlation with Z-spectrum

        Args:
            z_spectrum: (X, Y, Z, n_offsets) Z-spectrum data
            m0_candidates: Dict of {name: M0 array (X,Y,Z)}
            mask: (X, Y, Z) brain mask

        Returns:
            (selected_name, selected_m0, correlations_dict)
        """
        if len(m0_candidates) == 1:
            name, m0 = list(m0_candidates.items())[0]
            self.logger.info(f"  Only one M0 candidate: {name}")
            return name, m0, {name: 1.0}

        # Compute mean Z-spectrum intensity across offsets
        z_mean = z_spectrum.mean(axis=-1)  # (X, Y, Z)

        # Calculate correlation in brain mask
        correlations = {}
        for name, m0 in m0_candidates.items():
            # Extract brain voxels
            z_brain = z_mean[mask > 0]
            m0_brain = m0[mask > 0]

            # Compute Pearson correlation
            if len(z_brain) > 0 and len(m0_brain) > 0:
                corr = np.corrcoef(z_brain, m0_brain)[0, 1]
                correlations[name] = corr
            else:
                correlations[name] = -1.0

        # Select M0 with highest correlation
        best_name = max(correlations, key=correlations.get)
        best_corr = correlations[best_name]

        self.logger.info(f"  M0 selection based on correlation:")
        for name, corr in correlations.items():
            marker = " ← SELECTED" if name == best_name else ""
            self.logger.info(f"    {name}: r={corr:.4f}{marker}")

        return best_name, m0_candidates[best_name], correlations

    def check_if_normalized(self, z_spectrum: np.ndarray, mask: np.ndarray,
                           offsets_ppm: Optional[np.ndarray] = None) -> bool:
        """
        Check if Z-spectrum is already normalized (Z = S_sat / M0)

        Args:
            z_spectrum: (X, Y, Z, n_offsets) Z-spectrum data
            mask: (X, Y, Z) brain mask
            offsets_ppm: Optional frequency offsets in ppm (default: use config)

        Returns:
            True if normalized, False if raw S_sat
        """
        # Use config offsets if not provided
        if offsets_ppm is None:
            offsets_ppm = self.config.Z_SPECTRUM_OFFSETS_PPM

        n_offsets = z_spectrum.shape[-1]
        if len(offsets_ppm) != n_offsets:
            self.logger.warning(
                f"Offset count mismatch: {len(offsets_ppm)} vs {n_offsets}, "
                f"using default heuristic (first/last 10 channels)"
            )
            # Fallback to original heuristic
            far_indices = list(range(10)) + list(range(n_offsets - 10, n_offsets))
        else:
            # Use explicit offset-based selection: |Δω| > threshold
            threshold = self.config.FAR_OFFRESONANCE_THRESHOLD_PPM
            far_indices = np.where(np.abs(offsets_ppm) > threshold)[0].tolist()

            if len(far_indices) == 0:
                self.logger.warning(
                    f"No far-off-resonance channels found with |Δω| > {threshold} ppm, "
                    f"using first/last 10 channels"
                )
                far_indices = list(range(10)) + list(range(n_offsets - 10, n_offsets))

        # Extract far-off-resonance channels
        far_channels = z_spectrum[..., far_indices]

        # Calculate mean in brain
        brain_mean = far_channels[mask > 0].mean()

        is_normalized = 0.9 <= brain_mean <= 1.1

        self.logger.info(
            f"Z-spectrum far-off-resonance mean: {brain_mean:.4f} "
            f"(using {len(far_indices)} channels with |Δω| > {self.config.FAR_OFFRESONANCE_THRESHOLD_PPM} ppm) "
            f"→ {'normalized' if is_normalized else 'raw S_sat'}"
        )

        return is_normalized

    def process_z_spectrum_C(self,
                            z_spectrum: np.ndarray,
                            m0: np.ndarray,
                            mask: np.ndarray,
                            spacing_in: Tuple[float, float, float],
                            spacing_out: Tuple[float, float, float],
                            family: ChannelFamily,
                            b1_level: str = 'low',
                            z_offsets_ppm: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Method C for Z-spectrum: Re-normalize at low resolution

        Args:
            z_spectrum: (X, Y, Z, n_offsets) Z-spectrum data
            m0: (X, Y, Z) M0 reference
            mask: (X, Y, Z) brain mask
            spacing_in/out: Spacings in mm
            family: ChannelFamily for PSF matching
            b1_level: 'low' or 'high'
            z_offsets_ppm: Optional frequency offsets in ppm

        Returns:
            Dict with downsampled Z-spectrum and M0
        """
        self.logger.info(f"Processing Z-spectrum ({b1_level} B1) with Method C...")

        X, Y, Z, n_offsets = z_spectrum.shape

        # Check if normalized (pass z_offsets_ppm)
        is_normalized = self.check_if_normalized(z_spectrum, mask, z_offsets_ppm)

        if is_normalized:
            self.logger.info("  De-normalizing: S_sat = Z * M0")
            # De-normalize
            s_sat = z_spectrum * m0[..., np.newaxis]
        else:
            self.logger.info("  Already in S_sat format")
            s_sat = z_spectrum

        # Downsample S_sat (each offset independently)
        self.logger.info(f"  Downsampling {n_offsets} S_sat channels...")
        s_sat_lr_list = []
        for i in range(n_offsets):
            s_sat_lr = self.downsampler.downsample_family_D(
                s_sat[..., i], family, spacing_in, spacing_out, mask
            )
            s_sat_lr_list.append(s_sat_lr)

        s_sat_lr = np.stack(s_sat_lr_list, axis=-1)

        # Downsample M0
        self.logger.info("  Downsampling M0...")
        m0_lr = self.downsampler.downsample_family_D(
            m0, family, spacing_in, spacing_out, mask
        )

        # Re-normalize at low resolution
        self.logger.info("  Re-normalizing: Z' = S_sat' / M0'")

        # Numerical stability
        epsilon = 1e-10
        z_lr = s_sat_lr / (m0_lr[..., np.newaxis] + epsilon)

        # Clip to [0, 1] for training features (keep unclipped for fitting)
        z_lr_clipped = np.clip(z_lr, 0.0, 1.0)

        self.logger.info(f"  Z' range: [{z_lr.min():.4f}, {z_lr.max():.4f}]")
        self.logger.info(f"  Z' clipped range: [{z_lr_clipped.min():.4f}, {z_lr_clipped.max():.4f}]")

        return {
            'z_spectrum_lr': z_lr_clipped,
            'z_spectrum_lr_unclipped': z_lr,
            'm0_lr': m0_lr,
            's_sat_lr': s_sat_lr
        }


# ==============================================================================
# QA Metrics Computation
# ==============================================================================

class QAMetricsComputer:
    """Compute advanced QA metrics for downsampling validation"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def compute_nyquist_suppression_per_axis(self,
                                            data_hr: np.ndarray,
                                            data_lr: np.ndarray,
                                            spacing_in: Tuple[float, float, float],
                                            spacing_out: Tuple[float, float, float],
                                            mask_hr: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute high-frequency suppression near new Nyquist for each axis

        For each axis, compute energy ratio in the high-frequency band near the
        new Nyquist frequency (0.8-1.0 × f_nyquist_new) by upsampling LR back
        to HR resolution for same-domain comparison.

        Args:
            data_hr: High-resolution data (X, Y, Z)
            data_lr: Low-resolution data (X', Y', Z')
            spacing_in/out: Spacing in mm
            mask_hr: Optional brain mask

        Returns:
            Dict with suppression_db for x, y, z axes
        """
        # Apply mask if provided
        if mask_hr is not None:
            data_hr = data_hr * mask_hr

        # Upsample LR back to HR resolution for same-domain comparison
        image_lr = numpy_to_sitk(data_lr, spacing_out)

        # Calculate target size matching HR
        hr_size = list(data_hr.shape)

        # Resample LR to HR grid using linear interpolation
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([hr_size[0], hr_size[1], hr_size[2]])
        resampler.SetOutputSpacing(spacing_in)
        resampler.SetOutputOrigin(image_lr.GetOrigin())
        resampler.SetOutputDirection(image_lr.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)

        image_lr_upsampled = resampler.Execute(image_lr)
        data_lr_upsampled = sitk_to_numpy(image_lr_upsampled)

        # Compute FFT for both at HR resolution
        fft_hr = np.fft.fftn(data_hr)
        fft_lr_up = np.fft.fftn(data_lr_upsampled)

        power_hr = np.abs(fft_hr) ** 2
        power_lr_up = np.abs(fft_lr_up) ** 2

        results = {}

        # Process each axis independently
        for axis, axis_name in enumerate(['x', 'y', 'z']):
            # Get frequency array for this axis
            n = data_hr.shape[axis]
            freq = np.fft.fftfreq(n, d=spacing_in[axis])  # cycles/mm

            # New Nyquist frequency for this axis
            nyquist_new = 0.5 / spacing_out[axis]

            # Define high-frequency band: 0.8 to 1.0 × f_nyquist_new
            # This captures frequencies that should be suppressed by downsampling
            freq_band_low = 0.8 * nyquist_new
            freq_band_high = 1.0 * nyquist_new

            # Create mask for high-frequency band on this axis
            freq_mask = (np.abs(freq) >= freq_band_low) & (np.abs(freq) <= freq_band_high)

            # Expand mask to 3D by broadcasting
            if axis == 0:  # X axis
                freq_mask_3d = freq_mask[:, np.newaxis, np.newaxis]
            elif axis == 1:  # Y axis
                freq_mask_3d = freq_mask[np.newaxis, :, np.newaxis]
            else:  # Z axis
                freq_mask_3d = freq_mask[np.newaxis, np.newaxis, :]

            # Compute energy in high-frequency band for both HR and upsampled LR
            power_hr_band = np.sum(power_hr * freq_mask_3d)
            power_lr_up_band = np.sum(power_lr_up * freq_mask_3d)

            # Compute suppression ratio (same domain comparison)
            # suppression_db = 10 * log10(P_hr_band / P_lr_band)
            # Higher values mean better suppression
            if power_hr_band > 1e-10 and power_lr_up_band > 1e-10:
                suppression_db = 10 * np.log10(power_hr_band / (power_lr_up_band + 1e-10))
            else:
                suppression_db = 0.0

            # Ensure non-negative
            suppression_db = max(0.0, suppression_db)

            results[f'suppression_db_{axis_name}'] = float(suppression_db)

            self.logger.debug(
                f"    Axis {axis_name}: f_band=[{freq_band_low:.3f}, {freq_band_high:.3f}] cycles/mm, "
                f"P_hr={power_hr_band:.2e}, P_lr={power_lr_up_band:.2e}, "
                f"suppression={suppression_db:.2f} dB"
            )

        return results

    def compute_roundtrip_metrics(self,
                                 data_hr: np.ndarray,
                                 data_lr: np.ndarray,
                                 spacing_in: Tuple[float, float, float],
                                 spacing_out: Tuple[float, float, float],
                                 mask_hr: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute roundtrip fidelity: HR -> LR -> HR_recon

        Returns PSNR and SSIM metrics
        """
        # Upsample LR back to HR resolution using linear interpolation
        image_lr = numpy_to_sitk(data_lr, spacing_out)

        # Calculate target size
        original_size = list(data_hr.shape)

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([original_size[0], original_size[1], original_size[2]])
        resampler.SetOutputSpacing(spacing_in)
        resampler.SetOutputOrigin(image_lr.GetOrigin())
        resampler.SetOutputDirection(image_lr.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)

        image_hr_recon = resampler.Execute(image_lr)
        data_hr_recon = sitk_to_numpy(image_hr_recon)

        # Apply mask for fair comparison
        if mask_hr is not None:
            data_hr_masked = data_hr[mask_hr > 0]
            data_recon_masked = data_hr_recon[mask_hr > 0]
        else:
            data_hr_masked = data_hr.flatten()
            data_recon_masked = data_hr_recon.flatten()

        # Compute PSNR
        mse = np.mean((data_hr_masked - data_recon_masked) ** 2)
        if mse > 1e-10:
            # Assume data is in [0, 1] range for Z-spectra, or use max value
            data_range = max(data_hr_masked.max(), data_recon_masked.max())
            psnr_db = 10 * np.log10((data_range ** 2) / mse)
        else:
            psnr_db = 100.0  # Perfect reconstruction

        # Compute SSIM on full 3D volumes (use smaller window for speed)
        try:
            # Normalize to [0, 1] for SSIM
            hr_norm = (data_hr - data_hr.min()) / (data_hr.max() - data_hr.min() + 1e-10)
            recon_norm = (data_hr_recon - data_hr_recon.min()) / (data_hr_recon.max() - data_hr_recon.min() + 1e-10)

            # Use smaller window for 3D SSIM (default 7 might be too large)
            ssim_val = ssim(hr_norm, recon_norm, data_range=1.0, win_size=5)
        except Exception as e:
            self.logger.warning(f"SSIM computation failed: {e}")
            ssim_val = 0.0

        return {
            'roundtrip_psnr_db': float(psnr_db),
            'roundtrip_ssim': float(ssim_val)
        }


# ==============================================================================
# Main Pipeline
# ==============================================================================

class MRIDownsamplingPipeline:
    """
    Main pipeline for scientific downsampling of multi-modal MRI data
    """

    def __init__(self, output_dir: Path, log_level: str = 'INFO', random_seed: int = 42):
        """
        Initialize pipeline

        Args:
            output_dir: Output directory for logs and results
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.output_dir, log_level)
        self.logger.info("=" * 80)
        self.logger.info("MRI Multi-modal Downsampling Pipeline")
        self.logger.info("=" * 80)

        # Fix random seeds
        fix_random_seeds(random_seed)
        self.logger.info(f"Random seed fixed to: {random_seed}")

        # Initialize components
        self.config = ChannelConfig()
        self.axis_reorderer = AxisReorderer(self.logger)
        self.slab_localizer = SlabLocalizer(self.logger)
        self.downsampler = ChannelDownsampler(self.logger)
        self.z_processor = ZSpectrumProcessor(self.logger, self.downsampler)
        self.qa_computer = QAMetricsComputer(self.logger)

        # Metadata storage
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'random_seed': random_seed,
            'config': {
                'input_spacing_mm': self.config.INPUT_SPACING_MM,
                'target_spacing_mm': self.config.TARGET_SPACING_MM,
            }
        }

    def run(self,
            data: np.ndarray,
            region_mask: np.ndarray,
            region_labels: np.ndarray,
            align_to_128x104x18: bool = False,
            z_offsets_ppm: Optional[np.ndarray] = None,
            save_axis_order: str = "proc") -> Dict[str, Any]:
        """
        Run complete downsampling pipeline

        Args:
            data: (384, 336, 256, 351) input data in (Z, X, Y, C) order
            region_mask: (384, 336, 256) brain mask
            region_labels: (384, 336, 256) region labels (0-101)
            align_to_128x104x18: Whether to force align to this matrix size
            z_offsets_ppm: Optional array of Z-spectrum frequency offsets in ppm (54 points)
                          If None, uses default from config
            save_axis_order: Axis order for saved data {"proc", "orig"}
                           - "proc": Save in processing order (X,Y,Z,C) - default
                           - "orig": Save in original input order (Z,X,Y,C) with 1D reordering

        Returns:
            Dict containing:
                - data_lr: downsampled features (axis order depends on save_axis_order)
                - proba_labels: probability labels (axis order depends on save_axis_order)
                - region_mask_lr: downsampled mask (axis order depends on save_axis_order)
                - multidim_data: (n_voxels, 351) 1D features (if save_axis_order="orig")
                - seg_one_hot: (102, n_voxels) 1D labels (if save_axis_order="orig")
                - region_seg: (n_voxels,) 1D labels (if save_axis_order="orig")
                - metadata: Complete processing metadata
                - qa_metrics: QA metrics
        """
        self.logger.info(f"Input data shape: {data.shape}")
        self.logger.info(f"Expected format: (Z=384, X=336, Y=256, C=351)")

        # Store z_offsets_ppm in metadata
        if z_offsets_ppm is None:
            z_offsets_ppm = self.config.Z_SPECTRUM_OFFSETS_PPM
        else:
            z_offsets_ppm = np.asarray(z_offsets_ppm)

        self.metadata['z_spectrum_offsets_ppm'] = z_offsets_ppm.tolist()
        self.logger.info(f"Z-spectrum offsets: {len(z_offsets_ppm)} points from {z_offsets_ppm.min():.2f} to {z_offsets_ppm.max():.2f} ppm")

        # Step 0: Axis reordering
        step0_result = self.axis_reorderer.reorder_axes(data, region_mask, region_labels)
        data_xyz = step0_result['data']
        mask_xyz = step0_result['region_mask']
        labels_xyz = step0_result['region_labels']
        spacing_in = step0_result['input_spacing']

        # Step 1: CEST slab localization
        step1_result = self.slab_localizer.localize_and_crop(
            data_xyz, mask_xyz, labels_xyz, spacing_in
        )
        data_cropped = step1_result['data']
        mask_cropped = step1_result['region_mask']
        labels_cropped = step1_result['region_labels']

        # Store slab metadata
        self.metadata['slab_localization'] = {
            'bbox': step1_result['bbox'],
            'slab_thickness_mm': step1_result['slab_thickness_mm'],
            'slab_thickness_voxels': step1_result['slab_thickness_voxels'],
            'coverage_fallback': step1_result['coverage_fallback'],
            'thresholds': step1_result['thresholds'],
            'cropped_shape': step1_result['cropped_shape'],
            'physical_size_mm': step1_result['physical_size_mm']
        }

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Step 2-5: Channel-Family-Specific Downsampling")
        self.logger.info("=" * 80)

        # Prepare output arrays
        spacing_out = self.config.TARGET_SPACING_MM
        X_in, Y_in, Z_in, C = data_cropped.shape

        # Calculate output size
        X_out = int(np.round(X_in * spacing_in[0] / spacing_out[0]))
        Y_out = int(np.round(Y_in * spacing_in[1] / spacing_out[1]))
        Z_out = int(np.round(Z_in * spacing_in[2] / spacing_out[2]))

        self.logger.info(f"Calculated output size: ({X_out}, {Y_out}, {Z_out})")

        # Initialize output
        data_lr = np.zeros((X_out, Y_out, Z_out, 351), dtype=np.float32)

        # Track processing info
        channel_processing_log = []

        # Process each channel family
        for family_name, family in self.config.FAMILIES.items():
            self.logger.info(f"\nProcessing family: {family_name}")
            self.logger.info(f"  Channels (0-based): {len(family.indices)} channels")
            self.logger.info(f"  Native resolution: {family.native_resolution} mm")
            self.logger.info(f"  σ_add: {family.sigma_add_mm} mm")
            self.logger.info(f"  Method: {family.method}")
            self.logger.info(f"  Interpolator: {family.interpolator}")

            # Special handling for Z-spectra (Method C)
            if family_name == 'CEST':
                self._process_cest_family(
                    data_cropped, mask_cropped, data_lr,
                    spacing_in, spacing_out, family, z_offsets_ppm
                )
            else:
                # Standard Method D for other families
                for idx in family.indices:
                    channel_data = data_cropped[..., idx]

                    if family.method == 'D':
                        # Direct downsampling
                        data_lr[..., idx] = self.downsampler.downsample_family_D(
                            channel_data, family, spacing_in, spacing_out, mask_cropped
                        )
                    elif family.method == 'C':
                        # Method C: Recompute from upstream
                        # For now, fallback to Method D with warning
                        self.logger.warning(
                            f"  Channel {idx} ({family_name}): Method C not fully implemented, "
                            f"using Method D fallback"
                        )
                        data_lr[..., idx] = self.downsampler.downsample_family_D(
                            channel_data, family, spacing_in, spacing_out, mask_cropped
                        )

            # Log processing info
            channel_processing_log.append({
                'family': family_name,
                'channels': family.indices,
                'method': family.method,
                'sigma_add_mm': family.sigma_add_mm,
                'interpolator': family.interpolator
            })

        # Downsample mask (nearest neighbor)
        self.logger.info("\nDownsampling region mask...")
        mask_lr = self._downsample_mask(mask_cropped, spacing_in, spacing_out)

        # Generate probability labels
        self.logger.info("\nGenerating probability labels...")
        proba_labels = self._generate_probability_labels(
            labels_cropped, mask_cropped, spacing_in, spacing_out
        )

        # Store processing metadata
        self.metadata['channel_processing'] = channel_processing_log
        self.metadata['output_shape'] = {
            'data_lr': data_lr.shape,
            'mask_lr': mask_lr.shape,
            'proba_labels': proba_labels.shape
        }
        self.metadata['output_spacing_mm'] = spacing_out

        # Compute QA metrics
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Computing QA Metrics...")
        self.logger.info("=" * 80)

        qa_metrics = self._compute_qa_metrics(
            data_cropped, data_lr, mask_cropped, mask_lr,
            spacing_in, spacing_out
        )

        # Optional: Align to 128x104x18
        if align_to_128x104x18:
            self.logger.info("\nAligning to 128×104×18 matrix...")
            data_lr, mask_lr, proba_labels = self._align_to_target_size(
                data_lr, mask_lr, proba_labels, (128, 104, 18)
            )

        # Apply axis reordering and 1D conversion if requested
        result = {}
        if save_axis_order == "orig":
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Applying axis reordering to original (Z,X,Y,C) format")
            self.logger.info("=" * 80)

            result = self._apply_axis_reordering_for_save(
                data_lr, mask_lr, proba_labels, labels_cropped
            )

            # Add mapping metadata
            self.metadata['mapping'] = {
                'axes': 'input(Z,X,Y,C)->proc(X,Y,Z,C)->saved(Z,X,Y,C)',
                'permute': [1, 2, 0, 3],
                'inv_perm': [2, 0, 1],
                'save_axis_order': 'orig',
                'one_d_reorder_applied': True,
                'one_d_order_len': result.get('n_voxels', 0)
            }
        else:
            # Keep processing axis order
            result = {
                'data_lr': data_lr,
                'proba_labels': proba_labels,
                'region_mask_lr': mask_lr
            }

            self.metadata['mapping'] = {
                'axes': 'input(Z,X,Y,C)->proc(X,Y,Z,C)->saved(X,Y,Z,C)',
                'permute': [1, 2, 0, 3],
                'save_axis_order': 'proc',
                'one_d_reorder_applied': False
            }

        # Add metadata and QA metrics to result
        result['metadata'] = self.metadata
        result['qa_metrics'] = qa_metrics

        # Save metadata
        metadata_file = self.output_dir / 'pipeline_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        self.logger.info(f"\n✓ Pipeline completed successfully!")
        self.logger.info(f"  Save axis order: {save_axis_order}")
        if save_axis_order == "orig":
            self.logger.info(f"  3D data shape (Z,X,Y,C): {result['data_lr'].shape}")
            self.logger.info(f"  1D data shape: {result['multidim_data'].shape}")
        else:
            self.logger.info(f"  Output shape (X,Y,Z,C): {result['data_lr'].shape}")
        self.logger.info(f"  Metadata saved to: {metadata_file}")

        return result

    def _process_cest_family(self,
                            data: np.ndarray,
                            mask: np.ndarray,
                            data_lr: np.ndarray,
                            spacing_in: Tuple[float, float, float],
                            spacing_out: Tuple[float, float, float],
                            family: ChannelFamily,
                            z_offsets_ppm: np.ndarray):
        """Process CEST family with Z-spectrum Method C and automatic M0 selection"""

        # Process low B1 Z-spectrum
        z_low_indices = self.config.Z_SPECTRUM_LOW_B1
        z_low = data[..., z_low_indices]

        # For low B1, only one M0 option
        m0_low = data[..., self.config.M0_LOW_B1]

        self.logger.info("Processing low B1 Z-spectrum...")
        result_low = self.z_processor.process_z_spectrum_C(
            z_low, m0_low, mask, spacing_in, spacing_out, family, 'low', z_offsets_ppm
        )

        # Store results
        for i, idx in enumerate(z_low_indices):
            data_lr[..., idx] = result_low['z_spectrum_lr'][..., i]
        data_lr[..., self.config.M0_LOW_B1] = result_low['m0_lr']

        # Process high B1 Z-spectrum with automatic M0 selection
        z_high_indices = self.config.Z_SPECTRUM_HIGH_B1
        z_high = data[..., z_high_indices]

        # Collect all M0 candidates for high B1
        self.logger.info("Processing high B1 Z-spectrum...")
        m0_candidates = {
            'M0_HIGH_B1_PRIMARY': data[..., self.config.M0_HIGH_B1_PRIMARY],
            'M0_HIGH_B1_SECONDARY': data[..., self.config.M0_HIGH_B1_SECONDARY],
            'M0_FALLBACK': data[..., self.config.M0_FALLBACK]
        }

        # Automatically select best M0 based on correlation
        selected_m0_name, m0_high, m0_correlations = self.z_processor.select_best_m0(
            z_high, m0_candidates, mask
        )

        # Log selection and correlations to metadata
        if 'cest_m0_selection' not in self.metadata:
            self.metadata['cest_m0_selection'] = {}
        self.metadata['cest_m0_selection']['high_b1'] = {
            'selected': selected_m0_name,
            'correlations': {k: float(v) for k, v in m0_correlations.items()}
        }

        result_high = self.z_processor.process_z_spectrum_C(
            z_high, m0_high, mask, spacing_in, spacing_out, family, 'high', z_offsets_ppm
        )

        # Store results
        for i, idx in enumerate(z_high_indices):
            data_lr[..., idx] = result_high['z_spectrum_lr'][..., i]

        # Store the selected M0 in its primary position
        data_lr[..., self.config.M0_HIGH_B1_PRIMARY] = result_high['m0_lr']

        # Also downsample other M0 maps for consistency
        self.logger.info("  Downsampling additional M0 maps...")
        for m0_name, m0_data in m0_candidates.items():
            if m0_name != selected_m0_name:
                # Downsample but don't overwrite primary position
                if m0_name == 'M0_HIGH_B1_SECONDARY':
                    data_lr[..., self.config.M0_HIGH_B1_SECONDARY] = self.downsampler.downsample_family_D(
                        m0_data, family, spacing_in, spacing_out, mask
                    )
                elif m0_name == 'M0_FALLBACK':
                    data_lr[..., self.config.M0_FALLBACK] = self.downsampler.downsample_family_D(
                        m0_data, family, spacing_in, spacing_out, mask
                    )

    def _downsample_mask(self,
                        mask: np.ndarray,
                        spacing_in: Tuple[float, float, float],
                        spacing_out: Tuple[float, float, float]) -> np.ndarray:
        """Downsample binary mask using nearest neighbor"""
        image = numpy_to_sitk(mask.astype(np.float32), spacing_in)

        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        new_size = [
            int(np.round(original_size[i] * original_spacing[i] / spacing_out[i]))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(spacing_out)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0.0)

        resampled = resampler.Execute(image)
        result = sitk_to_numpy(resampled)

        return (result > 0.5).astype(np.uint8)

    def _generate_probability_labels(self,
                                    labels: np.ndarray,
                                    mask: np.ndarray,
                                    spacing_in: Tuple[float, float, float],
                                    spacing_out: Tuple[float, float, float]) -> np.ndarray:
        """
        Generate probability labels using MPRAGE->CEST PSF

        Args:
            labels: (X, Y, Z) integer labels 0-101
            mask: (X, Y, Z) brain mask
            spacing_in/out: Spacings in mm

        Returns:
            (X', Y', Z', 102) probability labels
        """
        # Use MPRAGE family PSF for label smoothing
        mprage_family = self.config.FAMILIES['MPRAGE']

        # Create a temporary family with MPRAGE PSF but LINEAR interpolator
        # IMPORTANT: Probability maps MUST use linear interpolation to avoid negative values
        # that would result from bspline overshooting on binary one-hot maps
        proba_family = ChannelFamily(
            name='MPRAGE_linear',
            indices=mprage_family.indices,
            native_resolution=mprage_family.native_resolution,
            sigma_add_mm=mprage_family.sigma_add_mm,
            method=mprage_family.method,
            interpolator='linear'  # Force linear instead of bspline
        )

        # Convert labels to one-hot
        num_classes = 102
        X, Y, Z = labels.shape
        onehot = np.zeros((X, Y, Z, num_classes), dtype=np.float32)

        # Create one-hot encoding
        for c in range(num_classes):
            onehot[..., c] = (labels == c).astype(np.float32)

        # Calculate output size
        X_out = int(np.round(X * spacing_in[0] / spacing_out[0]))
        Y_out = int(np.round(Y * spacing_in[1] / spacing_out[1]))
        Z_out = int(np.round(Z * spacing_in[2] / spacing_out[2]))

        proba_labels = np.zeros((X_out, Y_out, Z_out, num_classes), dtype=np.float32)

        self.logger.info(f"  Converting {num_classes} classes to probability maps...")
        self.logger.info(f"  Using MPRAGE PSF (σ={proba_family.sigma_add_mm} mm) with LINEAR interpolator")

        # Downsample each class probability map
        for c in range(num_classes):
            # Use proba_family with forced linear interpolation
            proba_labels[..., c] = self.downsampler.downsample_family_D(
                onehot[..., c], proba_family, spacing_in, spacing_out, mask
            )

        # Normalize probabilities to sum to 1
        prob_sum = proba_labels.sum(axis=-1, keepdims=True)
        prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)  # Avoid division by zero
        proba_labels = proba_labels / prob_sum

        # Clip to [0, 1]
        proba_labels = np.clip(proba_labels, 0.0, 1.0)

        self.logger.info(f"  Probability labels shape: {proba_labels.shape}")
        self.logger.info(f"  Sum check (should be ~1.0): {proba_labels.sum(axis=-1).mean():.6f}")

        return proba_labels

    def _compute_qa_metrics(self,
                           data_hr: np.ndarray,
                           data_lr: np.ndarray,
                           mask_hr: np.ndarray,
                           mask_lr: np.ndarray,
                           spacing_in: Tuple[float, float, float],
                           spacing_out: Tuple[float, float, float]) -> Dict[str, Any]:
        """Compute QA metrics including Nyquist suppression and roundtrip fidelity"""
        qa_metrics = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': data_hr.shape,
            'output_shape': data_lr.shape,
            'downsampling_ratio': tuple(
                s_out / s_in for s_in, s_out in zip(spacing_in, spacing_out)
            ),
            'brain_voxel_count': {
                'high_res': int(mask_hr.sum()),
                'low_res': int(mask_lr.sum())
            },
            'data_statistics': {},
            'frequency_domain_metrics': {},
            'roundtrip_fidelity': {}
        }

        # Compute per-family statistics
        for family_name, family in self.config.FAMILIES.items():
            if len(family.indices) == 0:
                continue

            # Sample first channel of family
            idx = family.indices[0]

            hr_data = data_hr[mask_hr > 0, idx]
            lr_data = data_lr[mask_lr > 0, idx]

            qa_metrics['data_statistics'][family_name] = {
                'channel_count': len(family.indices),
                'sample_channel': idx,
                'high_res': {
                    'mean': float(hr_data.mean()),
                    'std': float(hr_data.std()),
                    'min': float(hr_data.min()),
                    'max': float(hr_data.max())
                },
                'low_res': {
                    'mean': float(lr_data.mean()),
                    'std': float(lr_data.std()),
                    'min': float(lr_data.min()),
                    'max': float(lr_data.max())
                }
            }

            # Compute Nyquist suppression and roundtrip metrics for representative channels
            # Use MPRAGE as main reference (high SNR, isotropic native resolution)
            if family_name == 'MPRAGE':
                self.logger.info(f"  Computing advanced QA metrics for {family_name}...")

                hr_channel = data_hr[..., idx]
                lr_channel = data_lr[..., idx]

                # Nyquist suppression per axis
                nyquist_metrics = self.qa_computer.compute_nyquist_suppression_per_axis(
                    hr_channel, lr_channel, spacing_in, spacing_out, mask_hr
                )
                qa_metrics['frequency_domain_metrics'][family_name] = nyquist_metrics

                self.logger.info(f"    Nyquist suppression: "
                               f"X={nyquist_metrics['suppression_db_x']:.2f} dB, "
                               f"Y={nyquist_metrics['suppression_db_y']:.2f} dB, "
                               f"Z={nyquist_metrics['suppression_db_z']:.2f} dB")

                # Roundtrip fidelity
                roundtrip_metrics = self.qa_computer.compute_roundtrip_metrics(
                    hr_channel, lr_channel, spacing_in, spacing_out, mask_hr
                )
                qa_metrics['roundtrip_fidelity'][family_name] = roundtrip_metrics

                self.logger.info(f"    Roundtrip PSNR: {roundtrip_metrics['roundtrip_psnr_db']:.2f} dB, "
                               f"SSIM: {roundtrip_metrics['roundtrip_ssim']:.4f}")

        # Save QA report
        qa_file = self.output_dir / 'qa_report.json'
        with open(qa_file, 'w') as f:
            json.dump(qa_metrics, f, indent=2)

        self.logger.info(f"  QA report saved to: {qa_file}")

        return qa_metrics

    def _align_to_target_size(self,
                             data: np.ndarray,
                             mask: np.ndarray,
                             proba: np.ndarray,
                             target_size: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align arrays to target size by padding or cropping"""
        X, Y, Z, C = data.shape
        tX, tY, tZ = target_size

        # Pad or crop X
        if X < tX:
            pad_x = (tX - X) // 2
            data = np.pad(data, ((pad_x, tX - X - pad_x), (0, 0), (0, 0), (0, 0)))
            mask = np.pad(mask, ((pad_x, tX - X - pad_x), (0, 0), (0, 0)))
            proba = np.pad(proba, ((pad_x, tX - X - pad_x), (0, 0), (0, 0), (0, 0)))
        elif X > tX:
            crop_x = (X - tX) // 2
            data = data[crop_x:crop_x+tX, :, :, :]
            mask = mask[crop_x:crop_x+tX, :, :]
            proba = proba[crop_x:crop_x+tX, :, :, :]

        # Pad or crop Y
        if Y < tY:
            pad_y = (tY - Y) // 2
            data = np.pad(data, ((0, 0), (pad_y, tY - Y - pad_y), (0, 0), (0, 0)))
            mask = np.pad(mask, ((0, 0), (pad_y, tY - Y - pad_y), (0, 0)))
            proba = np.pad(proba, ((0, 0), (pad_y, tY - Y - pad_y), (0, 0), (0, 0)))
        elif Y > tY:
            crop_y = (Y - tY) // 2
            data = data[:, crop_y:crop_y+tY, :, :]
            mask = mask[:, crop_y:crop_y+tY, :]
            proba = proba[:, crop_y:crop_y+tY, :, :]

        # Pad or crop Z
        X, Y, Z, C = data.shape
        if Z < tZ:
            pad_z = (tZ - Z) // 2
            data = np.pad(data, ((0, 0), (0, 0), (pad_z, tZ - Z - pad_z), (0, 0)))
            mask = np.pad(mask, ((0, 0), (0, 0), (pad_z, tZ - Z - pad_z)))
            proba = np.pad(proba, ((0, 0), (0, 0), (pad_z, tZ - Z - pad_z), (0, 0)))
        elif Z > tZ:
            crop_z = (Z - tZ) // 2
            data = data[:, :, crop_z:crop_z+tZ, :]
            mask = mask[:, :, crop_z:crop_z+tZ]
            proba = proba[:, :, crop_z:crop_z+tZ, :]

        self.logger.info(f"  Aligned to {data.shape[:3]} (target: {target_size})")

        return data, mask, proba

    def _apply_axis_reordering_for_save(self,
                                       data_lr: np.ndarray,
                                       mask_lr: np.ndarray,
                                       proba_labels: np.ndarray,
                                       labels_hr: np.ndarray) -> Dict[str, Any]:
        """
        Reorder axes from proc (X,Y,Z,C) to original (Z,X,Y,C) and generate 1D data with reordering

        Args:
            data_lr: (X', Y', Z', 351) downsampled data in proc order
            mask_lr: (X', Y', Z') downsampled mask in proc order
            proba_labels: (X', Y', Z', 102) probability labels in proc order
            labels_hr: (X, Y, Z) high-res labels for generating region_seg

        Returns:
            Dict with 3D data in orig order + 1D data arrays
        """
        self.logger.info("Step S1: Inverting spatial axes to original order")

        # S1: Inverse permutation for 3D/4D arrays
        # inv_perm = (2, 0, 1) maps (X,Y,Z) -> (Z,X,Y)
        inv_perm_3d = (2, 0, 1)
        inv_perm_4d = (2, 0, 1, 3)

        data_save = np.transpose(data_lr, inv_perm_4d)  # (X',Y',Z',351) -> (Z',X',Y',351)
        mask_save = np.transpose(mask_lr, inv_perm_3d)  # (X',Y',Z') -> (Z',X',Y')
        proba_save = np.transpose(proba_labels, inv_perm_4d)  # (X',Y',Z',102) -> (Z',X',Y',102)

        self.logger.info(f"  data_lr: {data_lr.shape} -> {data_save.shape} (Z,X,Y,C)")
        self.logger.info(f"  mask_lr: {mask_lr.shape} -> {mask_save.shape} (Z,X,Y)")
        self.logger.info(f"  proba_labels: {proba_labels.shape} -> {proba_save.shape} (Z,X,Y,K)")

        # S2: Generate 1D data with row reordering
        self.logger.info("\nStep S2: Generating 1D data with reordering")

        # Check for empty ROI
        n_voxels_proc = int(mask_lr.sum())
        if n_voxels_proc == 0:
            raise ValueError("Empty ROI: region_mask_lr.sum() == 0, cannot generate 1D data")

        # Extract 1D data from proc order (ensures consistent ordering)
        multidim_data_proc = data_lr[mask_lr > 0]  # (n_voxels, 351)

        # Generate region_seg from labels (argmax of proba_labels)
        labels_lr = np.argmax(proba_labels, axis=-1).astype(np.uint8)  # (X',Y',Z')
        region_seg_proc = labels_lr[mask_lr > 0]  # (n_voxels,)

        # Generate seg_one_hot from proba_labels
        seg_one_hot_proc = proba_labels[mask_lr > 0].T  # (102, n_voxels)

        self.logger.info(f"  Extracted 1D data in proc order: n_voxels={n_voxels_proc}")

        # S2.1: Get linear indices in proc order
        idx_curr = np.flatnonzero(mask_lr.ravel(order="C"))

        # S2.2: Map to original axis coordinates
        coords_curr = np.column_stack(np.unravel_index(idx_curr, mask_lr.shape, order="C"))  # (n_voxels, 3) - (x,y,z)

        # Apply inverse permutation to coordinates: (x,y,z) -> (z,x,y)
        inv_perm_list = [2, 0, 1]
        coords_orig = coords_curr[:, inv_perm_list]  # (n_voxels, 3) - (z,x,y)

        # S2.3: Compute linear indices in saved axis order
        shape_orig = mask_save.shape  # (Z',X',Y')
        lin_orig = np.ravel_multi_index(coords_orig.T, shape_orig, order="C")

        # S2.4: Get reordering that matches saved C-order
        order = np.argsort(lin_orig)

        # S2.5: Apply reordering to 1D arrays
        multidim_data_save = multidim_data_proc[order, :]  # (n_voxels, 351)
        region_seg_save = region_seg_proc[order]  # (n_voxels,)
        seg_one_hot_save = seg_one_hot_proc[:, order]  # (102, n_voxels)

        self.logger.info(f"  Applied 1D reordering: {len(order)} voxels")
        self.logger.info(f"  multidim_data: {multidim_data_save.shape}")
        self.logger.info(f"  region_seg: {region_seg_save.shape}")
        self.logger.info(f"  seg_one_hot: {seg_one_hot_save.shape}")

        # S3: Validate 1D reordering (spot check)
        self.logger.info("\nStep S3: Validating 1D-3D correspondence")
        n_check = min(10, n_voxels_proc)
        check_indices = np.random.choice(n_voxels_proc, size=n_check, replace=False)

        max_diff = 0.0
        for i in check_indices:
            # Get saved 1D feature vector
            feat_1d = multidim_data_save[i, :]

            # Find corresponding 3D position in saved order
            saved_idx = np.where(mask_save.ravel(order="C"))[0][i]
            z, x, y = np.unravel_index(saved_idx, mask_save.shape, order="C")

            # Get 3D feature vector at same position
            feat_3d = data_save[z, x, y, :]

            diff = np.abs(feat_1d - feat_3d).max()
            max_diff = max(max_diff, diff)

        self.logger.info(f"  Spot check ({n_check} voxels): max_abs_diff = {max_diff:.2e}")
        if max_diff > 1e-5:
            self.logger.warning(f"  ⚠️  Large difference detected: {max_diff:.2e}")

        return {
            'data_lr': data_save,
            'region_mask_lr': mask_save,
            'proba_labels': proba_save,
            'multidim_data': multidim_data_save,
            'region_seg': region_seg_save,
            'seg_one_hot': seg_one_hot_save,
            'n_voxels': n_voxels_proc,
            'region': mask_save  # Alias for compatibility
        }


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == '__main__':
    """
    Example usage of the MRI Downsampling Pipeline
    """

    # Setup
    import scipy.io as sio

    # Load data (example)
    data_path = Path("/path/to/subject1_3d_validated.mat")

    if data_path.exists():
        print("Loading data...")
        mat_data = sio.loadmat(data_path)

        data = mat_data['data']  # (384, 336, 256, 351) in (Z, X, Y, C) order
        region_mask = mat_data['region_mask']  # (384, 336, 256)
        region_labels = mat_data['region_labels']  # (384, 336, 256)

        print(f"Data shape: {data.shape}")
        print(f"Mask shape: {region_mask.shape}")
        print(f"Labels shape: {region_labels.shape}")

        # Initialize pipeline
        output_dir = Path("./downsampling_output")
        pipeline = MRIDownsamplingPipeline(
            output_dir=output_dir,
            log_level='INFO',
            random_seed=42
        )

        # Run pipeline
        results = pipeline.run(
            data=data,
            region_mask=region_mask,
            region_labels=region_labels,
            align_to_128x104x18=False  # Set to True if you need exact size
        )

        # Access results
        data_lr = results['data_lr']
        proba_labels = results['proba_labels']
        mask_lr = results['region_mask_lr']

        print(f"\n✓ Downsampling completed!")
        print(f"  Output shape: {data_lr.shape}")
        print(f"  Probability labels shape: {proba_labels.shape}")
        print(f"  Mask shape: {mask_lr.shape}")

        # Save results
        output_file = output_dir / 'subject1_downsampled.npz'
        np.savez_compressed(
            output_file,
            data_lr=data_lr,
            proba_labels=proba_labels,
            region_mask_lr=mask_lr
        )
        print(f"  Results saved to: {output_file}")

    else:
        print(f"Data file not found: {data_path}")
        print("Please update the path in the script.")
