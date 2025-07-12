#!/usr/bin/env python3
"""
Reorder DWI NIfTI so that all b=0 volumes come first.
Inputs and outputs are passed from Snakemake:
  input.nifti, input.bval, input.bvec
  output.nifti, output.bval, output.bvec
"""

import nibabel as nib
import numpy as np

# Get input/output paths from Snakemake
nifti_in = snakemake.input.nifti     # noqa: F821
bval_in = snakemake.input.bval       # noqa: F821
bvec_in = snakemake.input.bvec       # noqa: F821
nifti_out = snakemake.output.nifti   # noqa: F821
bval_out = snakemake.output.bval     # noqa: F821
bvec_out = snakemake.output.bvec     # noqa: F821

# Load data
nifti = nib.load(nifti_in)
data = nifti.get_fdata()
bvals = np.loadtxt(bval_in)
bvecs = np.loadtxt(bvec_in)

# Handle case when bvecs is shape (N,) or (N,3) instead of (3, N)
if bvecs.ndim == 1:
    bvecs = bvecs.reshape((3, -1))
elif bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
    bvecs = bvecs.T

# Find b=0 indices (threshold < 50 is common)
b0_idx = np.where(bvals < 50)[0]
dwi_idx = np.where(bvals >= 50)[0]

# New order: b0 first
new_order = np.concatenate([b0_idx, dwi_idx])
new_data = data[..., new_order]
new_bvals = bvals[new_order]
new_bvecs = bvecs[:, new_order]

# Save reordered data
new_img = nib.Nifti1Image(new_data, nifti.affine, nifti.header)
nib.save(new_img, nifti_out)
np.savetxt(bval_out, new_bvals, fmt="%.2f")
np.savetxt(bvec_out, new_bvecs, fmt="%.6f")

print(f"Reordering complete: saved to {nifti_out}")
