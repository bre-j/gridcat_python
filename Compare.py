import io
import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.plotting as plotting
import pycircstat

from scipy.stats import circmean, circstd, pearsonr
from scipy.linalg import block_diag

from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs, load_img, new_img_like, mean_img, smooth_img, math_img
from nilearn.plotting import plot_design_matrix, plot_img
from nilearn.masking import apply_mask



class Comp:
    def __init__(self):
        pass

    def compare_masks(self, base_path):
        """
        quick comparison of masks"""
        matlab_mask = nib.load(f"{base_path}Matlab_GridCat_Output/GLM1/mask.nii")
        python_mask = nib.load(f"{base_path}python_gridcat_mask_py.nii.gz")

        pm = np.array(python_mask.dataobj)
        mm = np.array(matlab_mask.dataobj)
        print("-- Sanity Check --")
        print(f"Correlation between the created Python and the Matlab Mask {np.corrcoef(mm.flatten(), pm.flatten())[0,1]:.2f}")

    def get_roi_slices(self, roi_mask_path_or_list, method='range', n_slices_around=1):
        """
        Determines which slices to plot based on an ROI mask.
        """

        if isinstance(roi_mask_path_or_list, str):
            paths = [roi_mask_path_or_list] 
        elif isinstance(roi_mask_path_or_list, list):
            paths = roi_mask_path_or_list
        else:
            raise TypeError("not str or list provided")

        combined_mask_data = None
        
        for i, path in enumerate(paths):

            mask_img = nib.load(path)
            
            if i == 0:
                combined_mask_data = np.zeros(mask_img.shape, dtype=bool)

            combined_mask_data |= mask_img.get_fdata().astype(bool)

            
        _, _, z_indices = np.where(combined_mask_data)


        if method == 'all':
            return sorted(list(np.unique(z_indices)))

        elif method in ['center', 'range']:
            voxels_per_slice = np.sum(combined_mask_data, axis=(0, 1))
            center_slice = np.argmax(voxels_per_slice)

            if method == 'center':
                return [int(center_slice)]

            if method == 'range':
                start_slice = max(0, center_slice - n_slices_around)
                end_slice = min(combined_mask_data.shape[2], center_slice + n_slices_around + 1)
                return list(range(start_slice, end_slice))
            

    def compare_files(self, matlab_path, python_run, contrast_name, glm1_models, python_mask, z=12):
    # Load MATLAB beta map for SIN regressor

        matlab_beta_sin_img = nib.load(matlab_path)
        matlab_beta_sin_data = np.squeeze(matlab_beta_sin_img.get_fdata())
        print(f"MATLAB beta map '{os.path.basename(matlab_path)}' loaded. Shape: {matlab_beta_sin_data.shape}")

        # Get Python beta map for SIN regressor
        python_model_run1 = glm1_models[python_run]

        python_beta_sin_img = python_model_run1.compute_contrast(contrast_name, output_type='effect_size')
        python_beta_sin_data = np.squeeze(python_beta_sin_img.get_fdata())
        python_beta_sin_data[python_beta_sin_data == 0.0] = np.nan
        print(f"Python beta map for '{contrast_name}' Run {python_run} computed. Shape: {python_beta_sin_data.shape}")

        # --- 2. Basic Checks ---
        if matlab_beta_sin_data.shape != python_beta_sin_data.shape:
            print(f"Error: Shape mismatch! MATLAB: {matlab_beta_sin_data.shape}, Python: {python_beta_sin_data.shape}")
        else:
            print("Shapes match.")

            # --- 3. Masking
            mask_img_obj = python_mask # From your earlier code
            mask_data = mask_img_obj.get_fdata().astype(bool)

            # Apply mask to get 1D arrays of in-mask voxels
            matlab_betas_in_mask = matlab_beta_sin_data[mask_data]
            python_betas_in_mask = python_beta_sin_data[mask_data]

            print(f"Number of voxels in mask: {np.sum(mask_data)}")
            print(f"Number of values extracted for MATLAB: {len(matlab_betas_in_mask)}")
            print(f"Number of values extracted for Python: {len(python_betas_in_mask)}")

            # Handle potential NaNs from Python output if any within the mask
            # For fair comparison, we'll consider only voxels where both are finite
            finite_mask_in_common = np.isfinite(matlab_betas_in_mask) & np.isfinite(python_betas_in_mask)
            matlab_betas_common = matlab_betas_in_mask[finite_mask_in_common]
            python_betas_common = python_betas_in_mask[finite_mask_in_common]

            print(f"Number of common finite voxels in mask: {len(matlab_betas_common)}")

            if len(matlab_betas_common) < 2:
                print("Not enough common finite voxels in mask to perform detailed comparison.")
            else:
                print(f" Comparison ({contrast_name} Betas, Run 1) ---")
                stats_matlab = {
                    'min': np.min(matlab_betas_common), 'max': np.max(matlab_betas_common),
                    'mean': np.mean(matlab_betas_common), 'std': np.std(matlab_betas_common)
                }
                stats_python = {
                    'min': np.min(python_betas_common), 'max': np.max(python_betas_common),
                    'mean': np.mean(python_betas_common), 'std': np.std(python_betas_common)
                }

                print(f"{'Statistic':<10} {'MATLAB':<15} {'Python':<15} {'Difference':<15}")
                for stat_name in ['min', 'max', 'mean', 'std']:
                    m_val = stats_matlab[stat_name]
                    p_val = stats_python[stat_name]
                    diff = p_val - m_val
                    print(f"{stat_name:<10} {m_val:<15.6f} {p_val:<15.6f} {diff:<15.6f}")

                # Correlation
                correlation = np.corrcoef(matlab_betas_common, python_betas_common)[0, 1]
                print(f"\nPearson Correlation: {correlation:.6f}")

                # Mean Absolute Difference & Max Absolute Difference
                abs_diff = np.abs(python_betas_common - matlab_betas_common)
                mean_abs_diff = np.mean(abs_diff)
                max_abs_diff = np.max(abs_diff)
                print(f"Mean Absolute Difference: {mean_abs_diff:.6f}")
                print(f"Max Absolute Difference: {max_abs_diff:.6f}")



                # Create a difference map (unmasked for full FOV visualization)
                # Set values outside the mask to NaN for cleaner difference map display if desired
                diff_map_data = python_beta_sin_data - matlab_beta_sin_data
                print(matlab_beta_sin_data.shape)
                # diff_map_data[~mask_data] = np.nan # Optional: NaN out outside mask

                # Choose a representative slice (if no z ist specified picks middle axial slice)
                slices_to_plot = z
                for slice_idx in slices_to_plot:
                    print(slice_idx)
                    plt.figure(figsize=(18, 6))
                    plt.suptitle(f"{contrast_name} Beta Comparison (Run {python_run}, Slice Z={slice_idx})", fontsize=16)

                    plt.subplot(1, 3, 1)
                    plt.imshow(matlab_beta_sin_data[:, :, slice_idx].T, cmap='cold_hot', aspect='auto', origin='lower')
                    plt.colorbar(label='Beta Value')
                    plt.title(f'MATLAB {contrast_name} Betas')
                    plt.xlabel('X'); plt.ylabel('Y')

                    plt.subplot(1, 3, 2)
                    plt.imshow(python_beta_sin_data[:, :, slice_idx].T, cmap='cold_hot', aspect='auto', origin='lower')
                    plt.colorbar(label='Beta Value')
                    plt.title(f'Python {contrast_name} Betas')
                    plt.xlabel('X'); plt.ylabel('Y')

                    plt.subplot(1, 3, 3)
                    plt.imshow(diff_map_data[:, :, slice_idx].T, cmap='coolwarm', aspect='auto', origin='lower')
                    plt.colorbar(label='Difference (Python - MATLAB)')
                    plt.title('Difference Map')
                    plt.xlabel('X'); plt.ylabel('Y')

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.show()


    def compare_amp(self, matlab_amplitude_file, python_amplitude_file, run, z=12):
        matlab_img = nib.load(matlab_amplitude_file)
        print(matlab_img.shape)
        matlab_data = np.squeeze(matlab_img.get_fdata())

        python_img = nib.load(python_amplitude_file)
        python_data = np.squeeze(python_img.get_fdata())
        python_data[python_data == 0.0] = np.nan


        print(f"Matlab data shape: {matlab_data.shape}")
        print(f"Python data shape: {python_data.shape}")

        matlab_stats = {
            'min': np.nanmin(matlab_data), 'max': np.nanmax(matlab_data),
            'mean': np.nanmean(matlab_data), 'std': np.nanstd(matlab_data)
        }
        python_stats = {
            'min': np.nanmin(python_data), 'max': np.nanmax(python_data),
            'mean': np.nanmean(python_data), 'std': np.nanstd(python_data)
        }

        print(f"{'Statistic':<10} {'Matlab':<15} {'Python':<15} {'Difference':<15} {'Percent Diff':<15}")
        for stat in ['min', 'max', 'mean', 'std']:
            mat_stat = matlab_stats[stat]
            py_stat = python_stats[stat]
            diff = py_stat - mat_stat
            percent_diff = (diff / mat_stat) * 100 if mat_stat != 0 and not np.isnan(mat_stat) else np.inf
            print(f"{stat:<10} {mat_stat:<15.6f} {py_stat:<15.6f} {diff:<15.6f} {percent_diff:<15.2f}%")

        # Calculate correlation between the two datasets
        valid_mask = ~(np.isnan(matlab_data) | np.isnan(python_data))
        if np.sum(valid_mask) < 2: # Need at least 2 valid points for correlation
            print("Correlation cannot be computed: Not enough overlapping non-NaN voxels.")
            correlation = np.nan
        else:
            correlation = np.corrcoef(matlab_data[valid_mask].flatten(), python_data[valid_mask].flatten())[0, 1]
        print(f"Correlation between Matlab and Python output: {correlation:.6f}")

        # Calculate absolute difference
        abs_diff = np.abs(matlab_data - python_data)
        mean_abs_diff = np.nanmean(abs_diff)
        max_abs_diff = np.nanmax(abs_diff)
        print(f"Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"Maximum absolute difference: {max_abs_diff:.6f}")

        #Plot slices
        slices_to_plot = z
        for slice_idx in slices_to_plot:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(matlab_data[:, :, slice_idx], cmap='viridis')
            plt.colorbar(label='Amplitude'); plt.title('MATLAB Amplitude')
            plt.subplot(1, 2, 2)
            plt.imshow(python_data[:, :, slice_idx], cmap='viridis')
            plt.colorbar(label='Amplitude'); plt.title('Python Amplitude')
            plt.suptitle(f'{run} Amplitude Comparison - Slice {slice_idx}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            plt.show()
            plt.close()

    def circ_diff_deg(self, a, b):
        """
        Minimal absolute angular difference in degrees (circular)
        """
        diff = np.abs(a - b) % 360
        return np.where(diff > 180, 360 - diff, diff)



    def compare_ori(self, matlab_orientation_file, python_orientation_file, run, z=12):

        print("Comparing Python GLM1 Orientation with MATLAB")
        matlab_img = nib.load(matlab_orientation_file)
        matlab_data = np.squeeze(matlab_img.get_fdata())

        python_img = nib.load(python_orientation_file)
        python_data = np.squeeze(python_img.get_fdata())
        python_data[python_data == 0.0] = np.nan

        if matlab_data.shape != python_data.shape:
            print(f"Error: Shape mismatch! Matlab: {matlab_data.shape}, Python: {python_data.shape}")
        else:
            print(f"Matlab data shape: {matlab_data.shape}")
            print(f"Python data shape: {python_data.shape}")

            # --- Valid voxel mask ---
            valid_mask_ori = ~(np.isnan(matlab_data) | np.isnan(python_data))
            
            # --- Orientation stats ---
            matlab_data_f = matlab_data[valid_mask_ori]
            python_data_f = python_data[valid_mask_ori]
            print(f"Matlab data shape: {matlab_data.shape}")
            print(f"Python data shape: {python_data.shape}")
            
            matlab_stats = {
                'min': np.nanmin(matlab_data), 'max': np.nanmax(matlab_data),
                'mean': np.nanmean(matlab_data), 'std': np.nanstd(matlab_data)
            }
            python_stats = {
                'min': np.nanmin(python_data), 'max': np.nanmax(python_data),
                'mean': np.nanmean(python_data), 'std': np.nanstd(python_data)
            }

            print(f"{'Statistic':<10} {'Matlab':<15} {'Python':<15} {'Difference':<15} {'Percent Diff':<15}")
            for stat in ['min', 'max', 'mean', 'std']:
                mat_stat = matlab_stats[stat]
                py_stat = python_stats[stat]
                diff = py_stat - mat_stat
                percent_diff = (diff / mat_stat) * 100 if mat_stat != 0 and not np.isnan(mat_stat) else np.inf
                print(f"{stat:<10} {mat_stat:<15.6f} {py_stat:<15.6f} {diff:<15.6f} {percent_diff:<15.2f}%")


            correlation = np.corrcoef(matlab_data.flatten(), python_data.flatten())[0, 1]
            print(f"Correlation between Matlab and Python output: {correlation:.6f}")


            ori_diff = self.circ_diff_deg(matlab_data_f, python_data_f)

            mean_ori_mat = circmean(matlab_data_f, high=60, low=0)
            mean_ori_py = circmean(python_data_f, high=60, low=0)
            std_ori_mat = circstd(matlab_data_f, high=60, low=0)
            std_ori_py = circstd(python_data_f, high=60, low=0)
            mean_diff = np.nanmean(ori_diff)

            print(f"Mean Circular Orientation (MATLAB): {mean_ori_mat:.2f}°, Circular SD: {std_ori_mat:.2f}°")
            print(f"Mean Circular Orientation (Python): {mean_ori_py:.2f}°, Circular SD: {std_ori_py:.2f}°")
            print(f"Mean circular difference (MAT vs PY): {mean_diff:.2f}°")

            # Calculate absolute difference
            abs_diff = np.abs(matlab_data - python_data)
            mean_abs_diff = np.nanmean(abs_diff)
            max_abs_diff = np.nanmax(abs_diff)
            print(f"Mean absolute difference: {mean_abs_diff:.6f}")
            print(f"Maximum absolute difference: {max_abs_diff:.6f}")

        slices_to_plot = z

        for slice_idx in slices_to_plot:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(matlab_data[:, :, slice_idx], cmap='viridis')
            plt.colorbar(label='Orientation'); plt.title('MATLAB Orientation')
            plt.subplot(1, 2, 2)
            plt.imshow(python_data[:, :, slice_idx], cmap='viridis')
            plt.colorbar(label='Orientation'); plt.title('Python Orientation')
            plt.suptitle(f'Orientation {run}- Slice {slice_idx}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            plt.show()
            plt.close()


        plt.figure(figsize=(10, 4))
        plt.hist(matlab_data_f, bins=36, alpha=0.6, label='MATLAB')
        plt.hist(python_data_f, bins=36, alpha=0.6, label='Python')
        plt.axvline(np.nanmean(matlab_data), color='blue', linestyle='--', label=f'MAT mean: {np.nanmean(matlab_data_f):.1f}°')
        plt.axvline(np.nanmean(python_data), color='orange', linestyle='--', label=f'PY mean: {np.nanmean(python_data_f):.1f}°')
        plt.xlabel("Grid Orientation (°)")
        plt.ylabel("Voxel Count")
        plt.title(f"Grid Orientation Distribution ")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_beta_space(self, matlab_cos_path, matlab_sin_path, 
                           python_cos_img, python_sin_img, 
                           roi_mask_path, title="Beta Space Comparison"):
        """
        Compares MATLAB and Python GLM results in "beta space" for a given ROI.

        """
        print(f"\n--- Generating Beta Space Comparison for: {title} ---")

        try:
            matlab_betas_cos = apply_mask(matlab_cos_path, roi_mask_path)
            matlab_betas_sin = apply_mask(matlab_sin_path, roi_mask_path)
            python_betas_cos = apply_mask(python_cos_img, roi_mask_path)
            python_betas_sin = apply_mask(python_sin_img, roi_mask_path)
        except Exception as e:
            print(f"Error applying mask: {e}")
            return

        # 2. Calculate Mean Betas for the vectors
        mean_matlab_cos = np.mean(matlab_betas_cos)
        mean_matlab_sin = np.mean(matlab_betas_sin)
        mean_python_cos = np.mean(python_betas_cos)
        mean_python_sin = np.mean(python_betas_sin)

        # 3. Quantitative Comparison (Angle and Magnitude)
        # Calculate angle (in degrees) using arctan2 for quadrant correctness
        angle_matlab = np.rad2deg(np.arctan2(mean_matlab_sin, mean_matlab_cos))
        angle_python = np.rad2deg(np.arctan2(mean_python_sin, mean_python_cos))
        
        # Calculate magnitude (vector length)
        mag_matlab = np.sqrt(mean_matlab_cos**2 + mean_matlab_sin**2)
        mag_python = np.sqrt(mean_python_cos**2 + mean_python_sin**2)

        print("Mean Vector Analysis:")
        print(f"{'Pipeline':<10} {'Angle (°)':<15} {'Magnitude':<15}")
        print("-" * 40)
        print(f"{'MATLAB':<10} {angle_matlab:<15.2f} {mag_matlab:<15.4f}")
        print(f"{'Python':<10} {angle_python:<15.2f} {mag_python:<15.4f}")
        print("-" * 40)
        print(f"Angle Difference: {self.circ_diff_deg(angle_matlab, angle_python):.2f}°")
        
        # 4. Create the Visualization
        plt.figure(figsize=(8, 8))
        
        # Scatter plots for individual voxels
        plt.scatter(matlab_betas_cos, matlab_betas_sin, alpha=0.4, color='crimson', label='MATLAB Voxels')
        plt.scatter(python_betas_cos, python_betas_sin, alpha=0.4, color='royalblue', label='Python Voxels')
        
        # Plot the mean vectors as arrows
        plt.annotate('', xy=(mean_matlab_cos, mean_matlab_sin), xytext=(0, 0),
                     arrowprops=dict(facecolor='crimson', shrink=0, width=2, headwidth=10))
        plt.annotate('', xy=(mean_python_cos, mean_python_sin), xytext=(0, 0),
                     arrowprops=dict(facecolor='royalblue', shrink=0, width=2, headwidth=10))

        # Plot aesthetics
        plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
        plt.grid(True, linestyle=':')
        plt.xlabel('Beta Weight for Cosine Regressor')
        plt.ylabel('Beta Weight for Sine Regressor')
        plt.title(title, fontsize=14)
        plt.gca().set_aspect('equal', adjustable='box') # ESSENTIAL for correct angle perception

        # Create custom legend entries for the arrows
        plt.plot([], [], color='crimson', marker='>', linestyle='None', markersize=10, label='MATLAB Mean Vector')
        plt.plot([], [], color='royalblue', marker='>', linestyle='None', markersize=10, label='Python Mean Vector')
        plt.legend()
        plt.tight_layout()
        plt.show()


    def compare_glm2_maps(self, matlab_map_path, python_map_obj, mask_img_obj,
                        map_type_name="GLM2 Contrast", comparison_label="GLM2"):
        """
        Compares MATLAB and Python output maps for GLM2.
        """
        # Load MATLAB contrast map
        matlab_img = nib.load(matlab_map_path)
        matlab_data = np.squeeze(matlab_img.get_fdata(dtype=np.float32))
        if matlab_data.ndim > 3:
            matlab_data = matlab_data[..., 0]
        
        # Load Python contrast map data
        python_data = np.squeeze(python_map_obj.get_fdata(dtype=np.float32))
        if python_data.ndim > 3:
            python_data = python_data[..., 0]
        # Replace exact zero values with NaN for fair comparison
        python_data[python_data == 0] = np.nan
        
        # Check if shapes match
        if matlab_data.shape != python_data.shape:
            print(f"Warning: Shape mismatch! MATLAB: {matlab_data.shape}, Python: {python_data.shape}")
        else:
            print("Shapes match.")
        
        # Load the mask and apply it
        mask_data = np.squeeze(mask_img_obj.get_fdata()).astype(bool)
        
        # Apply the mask
        matlab_masked = matlab_data[mask_data]
        python_masked = python_data[mask_data]
        
        # Compute statistics on common finite values
        finite_mask = np.isfinite(matlab_masked) & np.isfinite(python_masked)
        matlab_common = matlab_masked[finite_mask]
        python_common = python_masked[finite_mask]
        if matlab_common.size < 2:
            print("Not enough finite voxels in common for statistics.")
        else:
            print("Statistics for voxels within the mask:")
            for stat in ['min', 'max', 'mean', 'std']:
                m_val = getattr(np, stat)(matlab_common)
                p_val = getattr(np, stat)(python_common)
                diff = p_val - m_val
                print(f"{stat.capitalize()}: MATLAB = {m_val:.6f}, Python = {p_val:.6f}, Diff = {diff:.6f}")
            corr = np.corrcoef(matlab_common, python_common)[0, 1]
            print(f"Pearson Correlation: {corr:.6f}")
        
        # Plotting a representative axial slice (middle slice of z-dimension)
        slice_idx = matlab_data.shape[2] // 2
        diff_map = python_data - matlab_data
        
        plt.figure(figsize=(18, 6))
        plt.suptitle(f"{comparison_label}: {map_type_name} Comparison (Slice {slice_idx})", fontsize=16)
        
        plt.subplot(1, 3, 1)
        plt.imshow(matlab_data[:, :, slice_idx].T, cmap='cold_hot', aspect='auto', origin='lower')
        plt.title("MATLAB Contrast")
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(python_data[:, :, slice_idx].T, cmap='cold_hot', aspect='auto', origin='lower')
        plt.title("Python Contrast")
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(diff_map[:, :, slice_idx].T, cmap='coolwarm', aspect='auto', origin='lower')
        plt.title("Difference (Python - MATLAB)")
        plt.colorbar()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    def compare_glm2_maps_in_roi(self, matlab_map_path, python_map_obj, roi_mask_obj,
                                 map_type_name="GLM2 Contrast", comparison_label="GLM2"):
        """
        Compares GLM2 contrast maps and visualizes them correctly within an ROI.
        
        Args:
            matlab_map_path (str): Path to MATLAB GLM2 contrast map.
            python_map_obj (Nifti1Image): Python GLM2 contrast map object.
            roi_mask_obj (Nifti1Image): The ANATOMICAL ROI MASK for this analysis.
            ...
        """
        matlab_img = nib.load(matlab_map_path)
        matlab_data = np.squeeze(matlab_img.get_fdata(dtype=np.float32))
        if matlab_data.ndim > 3:
            matlab_data = matlab_data[..., 0]
        
        # Load Python contrast map data
        python_data = np.squeeze(python_map_obj.get_fdata(dtype=np.float32))
        if python_data.ndim > 3:
            python_data = python_data[..., 0]
        
        # NOTE: I'm removing the replacement of zeros with NaN here.
        # It's better to do that on the masked data for stats, but not
        # on the original data for visualization unless intended.
        
        # Check if shapes match
        if matlab_data.shape != python_data.shape:
            print(f"Warning: Shape mismatch! MATLAB: {matlab_data.shape}, Python: {python_data.shape}")
        else:
            print("Shapes match.")
        
        # Load the mask and apply it for statistics
        mask_data = np.squeeze(roi_mask_obj.get_fdata()).astype(bool)
        matlab_masked = matlab_data[mask_data]
        python_masked = python_data[mask_data]
        
        # Make a copy for stats and replace 0 with NaN for fair comparison
        python_masked_for_stats = python_masked.copy()
        python_masked_for_stats[python_masked_for_stats == 0] = np.nan

        finite_mask = np.isfinite(matlab_masked) & np.isfinite(python_masked_for_stats)
        matlab_common = matlab_masked[finite_mask]
        python_common = python_masked_for_stats[finite_mask]
        
        if matlab_common.size < 2:
            print("Not enough finite voxels in common for statistics.")
        else:
            print("Statistics for voxels within the mask:")
            for stat in ['min', 'max', 'mean', 'std']:
                m_val = getattr(np, stat)(matlab_common)
                p_val = getattr(np, stat)(python_common)
                diff = p_val - m_val
                print(f"{stat.capitalize()}: MATLAB = {m_val:.6f}, Python = {p_val:.6f}, Diff = {diff:.6f}")
            corr = np.corrcoef(matlab_common, python_common)[0, 1]
            print(f"Pearson Correlation: {corr:.6f}")

        # --- 2. IMPROVED VISUALIZATION ---
        print("Visualizing GLM2 contrast within the provided ROI...")

        # Find a good coordinate to display by finding the center of mass of the ROI
        try:
            coords = plotting.find_xyz_cut_coords(roi_mask_obj)
        except (ValueError, IndexError): # Catch potential errors
            print("Could not find center of mass for ROI, using default coordinates.")
            coords = None
        

        python_display_data = np.zeros_like(python_data)
        python_display_data[mask_data] = python_data[mask_data]
        
        matlab_display_data = np.zeros_like(matlab_data)
        matlab_display_data[mask_data] = matlab_data[mask_data]

        python_map_in_roi = new_img_like(python_map_obj, python_display_data)
        matlab_img_in_roi = new_img_like(matlab_img, matlab_display_data)
        # Plot Python map within the ROI
        plotting.plot_stat_map(
            # Pass the NEW, MASKED Nifti object to the plotter
            stat_map_img=python_map_in_roi,
            bg_img=None, 
            display_mode='ortho',
            cut_coords=coords,
            title=f"Python: {map_type_name} within ROI",
            colorbar=True,
            threshold=1e-6 # Use a very small number to avoid plotting pure zeros
        )
        plt.show()

        # Plot MATLAB map within the ROI
        plotting.plot_stat_map(
            # Pass the NEW, MASKED Nifti object here too
            stat_map_img=matlab_img_in_roi,
            bg_img=None,
            display_mode='ortho',
            cut_coords=coords,
            title=f"MATLAB: {map_type_name} within ROI",
            colorbar=True,
            threshold=1e-6
        )
        plt.show()

    def compare_design_glm1(self, spm_csv_path, run_idx, glm1_models):
        """
        Compares a GLM1 Nilearn design matrix against an SPM/Matlab design matrix CSV.
        Plots regressors side-by-side and reports Pearson correlation, mean diff, R².
        """
        spm_dm_full = pd.read_csv(spm_csv_path)
        nilearn_dm_original = glm1_models[run_idx].design_matrices_[0]
        n_scans_current_run = nilearn_dm_original.shape[0]
        start_row = sum(
            glm1_models[i].design_matrices_[0].shape[0] for i in range(1, run_idx)
        )
        spm_dm_run = spm_dm_full.iloc[start_row:start_row + n_scans_current_run].copy()
        spm_dm_run.reset_index(drop=True, inplace=True)
        nilearn_dm_run = nilearn_dm_original.copy()

        print(spm_dm_run.columns)
        print("--- Starting Diagnostic Analysis ---")

        regressor_pairs = {
            "pmod_SIN": ("translation_pmodSIN", f"Sn({run_idx}) GridEvent-translationx-pmodSIN^1*bf(1)"),
            "pmod_COS": ("translation_pmodCOS", f"Sn({run_idx}) GridEvent-translationx-pmodCOS^1*bf(1)"),
            "main_translation": ("translation", f"Sn({run_idx}) GridEvent-translation*bf(1)"),
            "motion_X": ("X", f"Sn({run_idx}) R1"),
        }

        results = {}
        for name, (nilearn_col, spm_col) in regressor_pairs.items():
            print(f"\n--- Analyzing Regressor: {name} ---")
            nilearn_vec = nilearn_dm_run[nilearn_col].values
            spm_vec = spm_dm_run[spm_col].values

            corr, p_val = pearsonr(nilearn_vec, spm_vec)
            mean_diff = np.mean(nilearn_vec) - np.mean(spm_vec)
            std_ratio = np.std(nilearn_vec) / np.std(spm_vec)
            residuals = spm_vec - np.poly1d(np.polyfit(nilearn_vec, spm_vec, 1))(nilearn_vec)
            r_squared = 1 - (np.var(residuals) / np.var(spm_vec))

            print(f"  Pearson Correlation: {corr:.6f} (p={p_val:.4f})")
            print(f"  Mean Difference (Nilearn - SPM): {mean_diff:.6f}")
            print(f"  Std Dev Ratio (Nilearn / SPM): {std_ratio:.6f}")
            print(f"  R-squared: {r_squared:.6f}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(nilearn_vec, label=f"Nilearn: {nilearn_col}", linewidth=2.5, alpha=0.8)
            ax1.plot(spm_vec, label=f"SPM: {spm_col}", linestyle="--", linewidth=2.0)
            ax1.set_title(f"Comparison for: {name} | r={corr:.4f}", fontsize=16)
            ax1.set_ylabel("Regressor Value"); ax1.legend(); ax1.grid(True)
            ax2.plot(nilearn_vec - spm_vec)
            ax2.set_title("Difference (Nilearn - SPM)"); ax2.set_xlabel("Scan"); ax2.grid(True)
            plt.tight_layout(); plt.show()

            results[name] = {"correlation": corr, "mean_difference": mean_diff,
                             "std_dev_ratio": std_ratio, "r_squared": r_squared}

        print("\n\n--- FINAL DIAGNOSTIC SUMMARY ---")
        print(pd.DataFrame(results).T.to_string())

    def compare_glm_design_matrices(
        self,
        nilearn_dms,
        spm_csv_path,
        run_number,
        method,
        nilearn_aligned_col='translation_test_aligned',
        nilearn_misaligned_col='translation_test_misaligned',
        nilearn_main_translation_col='translation_test',
        nilearn_pmod_col='translation_test_pmodAlignment',
        nilearn_feedback_col='feedback_test',
        nilearn_motion_col='X'
    ):
        """
        Compares a GLM2 Nilearn design matrix against an SPM/Matlab design matrix CSV.
        Supports 'pmod' and 'aligned_misaligned' methods.
        Returns a summary DataFrame.
        """
        print(f"--- Diagnostic Analysis for Run {run_number} (Method: {method}) ---")
        nilearn_dm = nilearn_dms[run_number - 1]

        try:
            spm_dm_full = pd.read_csv(spm_csv_path)
        except FileNotFoundError:
            print(f"ERROR: SPM design matrix not found at: {spm_csv_path}")
            return pd.DataFrame()

        n_scans = nilearn_dm.shape[0]
        start_row = (run_number - 1) * n_scans
        spm_dm_run = spm_dm_full.iloc[start_row:start_row + n_scans].copy()

        if method == 'aligned_misaligned':
            regressor_pairs = {
                'aligned_events': (nilearn_aligned_col,
                    f'Sn({run_number}) GridEvent-translation-alignedWithROI1meanOri*bf(1)'),
                'misaligned_events': (nilearn_misaligned_col,
                    f'Sn({run_number}) GridEvent-translation-misalignedWithROI1meanOri*bf(1)')
            }
        elif method == 'pmod':
            regressor_pairs = {
                'main_translation_event': (nilearn_main_translation_col,
                    f'Sn({run_number}) GridEvent-translation*bf(1)'),
                'alignment_pmod': (nilearn_pmod_col,
                    f'Sn({run_number}) GridEvent-translationx-PMOD-alignmentWithROI1meanOri^1*bf(1)')
            }
        else:
            raise ValueError(f"Invalid method '{method}'. Choose 'aligned_misaligned' or 'pmod'.")

        regressor_pairs.update({
            'feedback_events': (nilearn_feedback_col, f'Sn({run_number}) feedback*bf(1)'),
            'motion_X': (nilearn_motion_col, f'Sn({run_number}) R1')
        })

        results = {}
        for name, (nilearn_col, spm_col) in regressor_pairs.items():
            print(f"\n--- Regressor: {name} ---")
            if nilearn_col not in nilearn_dm.columns or spm_col not in spm_dm_run.columns:
                print(f"  SKIPPING: column not found (Nilearn: '{nilearn_col}', SPM: '{spm_col}').")
                continue

            nilearn_vec = nilearn_dm[nilearn_col].values
            spm_vec = spm_dm_run[spm_col].values

            corr, p_val = pearsonr(nilearn_vec, spm_vec)
            mean_diff = np.mean(nilearn_vec) - np.mean(spm_vec)
            std_ratio = np.std(nilearn_vec) / np.std(spm_vec) if np.std(spm_vec) != 0 else float('inf')

            print(f"  Pearson Correlation: {corr:.6f} (p={p_val:.4f})")
            print(f"  Mean Difference (Nilearn - SPM): {mean_diff:.6f}")
            print(f"  Std Dev Ratio (Nilearn / SPM): {std_ratio:.6f}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(nilearn_vec, label=f'Nilearn: {nilearn_col}', alpha=0.8)
            ax1.plot(spm_vec, label=f'SPM: {spm_col}', linestyle='--')
            ax1.set_title(f'Run {run_number} - {name} | r={corr:.4f}', fontsize=16)
            ax1.set_ylabel('Regressor Value'); ax1.legend(); ax1.grid(True)
            ax2.plot(nilearn_vec - spm_vec)
            ax2.set_title('Difference (Nilearn - SPM)'); ax2.set_xlabel('Scan'); ax2.grid(True)
            plt.tight_layout(); plt.show()

            results[name] = {'correlation': corr, 'mean_difference': mean_diff, 'std_dev_ratio': std_ratio}

        summary_df = pd.DataFrame(results).T
        print(f"\n\n--- FINAL SUMMARY FOR RUN {run_number} (Method: {method}) ---")
        print(summary_df.to_string() if not summary_df.empty else "No results to display.")
        return summary_df

    def parse_and_print_metrics_file(self, filepath):
        """
        Parses and prints the semicolon-separated GridCAT metrics output file.
        """
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return

        blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
        for i, block_str in enumerate(blocks):
            df = pd.read_csv(io.StringIO(block_str), sep=';')
            metric_name = df.columns[0]
            print(f"--- Metric Section {i + 1}: {metric_name} ---\n")
            print(df)