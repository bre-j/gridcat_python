import os
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix

import nibabel as nib
from nilearn.image import (
    concat_imgs,
    load_img,
    new_img_like,
)
from nilearn.masking import apply_mask
from nilearn.glm.first_level import FirstLevelModel

import pycircstat

# Custom module for comparing to Matlab Output
import Compare


def event_use(strat):
    """
    Convert an event-splitting strategy name into GLM1 and GLM2 usage codes.
    """
    strat = strat.lower()
    if strat in ["half", "half_first"]:
        return 2, 3
    elif strat == "half_second":
        return 3, 2
    elif strat == "odd_first":
        return 4, 5
    elif strat == "even_first":
        return 5, 4
    elif strat == "all":
        return 8, None
    elif strat == "table":
        return 9, None
    else:
        raise ValueError(
            f"Unknown strategy '{strat}'. Must be one of: half, half_first, half_second, "
            "odd_first, even_first, all, table"
        )


class gridcat:
    """
    Main GridCAT Python workflow helper.

    Copyright 2017 Matthias Stangl, Jonathan Shine, Thomas Wolbers

    This file is part of the GridCAT Python Translation.

    See the GNU General Public License for more details.
    """

    def __init__(self, base_path):
        """Store the project base path and initialize comparison helpers."""
        self.base_path = base_path
        self.com = Compare.Comp()

    def create_spm_mask(self, run_folders, masking_threshold_fraction, mask_filename, matlab_compare=True):
        """
        Create an SPM-style analysis mask from all functional scans.
        Optionally compares the generated mask with the Matlab/GridCAT mask.
        """
        all_func_files = []
        for folder in run_folders:
            nii_files = sorted(glob.glob(os.path.join(folder, "*.nii")))
            niigz_files = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
            run_files = nii_files + niigz_files
            all_func_files.extend(run_files)

        first_img = None
        ref_header = None
        ref_affine = None
        sum_img_data = None
        n_volumes = 0

        for i, func_file in enumerate(all_func_files):

            img = nib.load(func_file)

            if first_img is None:
                first_img = img
                ref_header = img.header
                ref_affine = img.affine
                sum_img_data = np.zeros(img.shape, dtype=np.float64)
                print(f"Reference image shape set to: {sum_img_data.shape}")

            vol_data = img.get_fdata(dtype=np.float64)
            sum_img_data += vol_data
            n_volumes += 1

        mean_image_data = sum_img_data / n_volumes
        print(f"Calculated mean image from {n_volumes} processed 3D files.")

        global_mean_intensity = np.mean(mean_image_data)
        print(f"Global mean intensity of the mean image: {global_mean_intensity:.4f}")

        intensity_threshold = global_mean_intensity * masking_threshold_fraction
        print(f"Intensity threshold for mask: {intensity_threshold:.4f}")

        mask_data_boolean = mean_image_data > intensity_threshold
        mask_data_int = mask_data_boolean.astype(np.int8)

        n_mask_voxels = np.sum(mask_data_int)
        print(f"Mask includes {n_mask_voxels} voxels.")

        mask_img = nib.Nifti1Image(mask_data_int, ref_affine, ref_header)
        mask_img.set_data_dtype(np.int8)

        nib.save(mask_img, os.path.join(self.base_path, mask_filename))

        if matlab_compare:
            self.com.compare_masks(self.base_path)

        return mask_img

    def read_event_table_to_df(self, event_table_file, columns_provided=0):
        """
        Read an event table into a sorted DataFrame.
        Numeric timing and orientation columns are coerced for downstream GLMs.
        """
        if columns_provided != type(list):
            column_names = ["trial_type", "onset", "duration", "orientation"]

        else:
            column_names = columns_provided

        try:
            df = pd.read_csv(event_table_file, sep=";|\t|,", engine="python", header=None, names=column_names)
            df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
            df["orientation"] = pd.to_numeric(df["orientation"], errors="coerce")
            df = df.sort_values(by="onset").reset_index(drop=True)
            print(f"Read {len(df)} events from {os.path.basename(event_table_file)}")
            return df
        except Exception as e:
            print(f"Error reading {event_table_file}: {e}")
            return None

    def read_additional_regressors_to_df(self, regressor_file):
        """
        Read additional motion regressors into a DataFrame.
        The expected columns are X, Y, Z, Yaw, Pitch, and Roll.
        """

        column_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll"]
        try:
            df = pd.read_csv(regressor_file, sep=r"\s+|\t", engine="python", header=None, names=column_names)
            print(
                f"Read {len(df)} timepoints for {len(df.columns)} additional regressors "
                f"from {os.path.basename(regressor_file)}"
            )
            return df
        except Exception as e:
            print(f"Error reading {regressor_file}: {e}")
            return None


    def select_events_for_glm(self, event_df, usage_specifier, output_column_name, grid_event_label="translation"):
        """
        Mark grid events for inclusion in a GLM based on a usage rule.
        Returns a copy of the event table with a boolean selection column.
        """
        df = event_df.copy()
        # Ensure DataFrame is sorted by onset for consistent splitting
        df = df.sort_values(by="onset").reset_index(drop=True)

        # Initialize the output column to False for all events initially
        df[output_column_name] = False

        # Identify the specific grid events that have valid orientation data
        grid_event_mask = (df["trial_type"] == grid_event_label) & (~df["orientation"].isna())
        grid_event_indices = df.index[grid_event_mask]
        n_grid_events = len(grid_event_indices)

        if n_grid_events == 0:
            print(f"Warning: No events found matching trial_type='{grid_event_label}' with valid orientation.")
            return df

        print(f"Found {n_grid_events} '{grid_event_label}' events with orientation.")
        print(f"Applying selection rule: {usage_specifier}")

        indices_to_use = []  # List to store the DataFrame indices to mark True

        if usage_specifier == 1:  # Use ALL grid events
            indices_to_use = grid_event_indices
            print(f"  Rule {usage_specifier}: Selecting all {len(indices_to_use)} grid events.")

        elif usage_specifier == 2:  # Use FIRST HALF
            # This uses Matlab-style indexing, which was a previous source of mismatch.
            half_point = int(np.floor(n_grid_events / 2 + 0.5))
            indices_to_use = grid_event_indices[:half_point]
            print(
                f"  Rule {usage_specifier}: Selecting first half - "
                f"{len(indices_to_use)} of {n_grid_events} grid events."
            )

        elif usage_specifier == 3:  # Use SECOND HALF
            half_point = int(np.floor(n_grid_events / 2 + 0.5))
            indices_to_use = grid_event_indices[half_point:]
            print(
                f"  Rule {usage_specifier}: Selecting second half - "
                f"{len(indices_to_use)} of {n_grid_events} grid events."
            )

        elif usage_specifier == 4:  # Use ODD events.
            indices_to_use = grid_event_indices[::2]  # Python slicing [start:stop:step]
            print(
                f"  Rule {usage_specifier}: Selecting odd-indexed - "
                f"{len(indices_to_use)} of {n_grid_events} grid events."
            )

        elif usage_specifier == 5:  # Use EVEN events (index 1, 3, 5... in the grid-event list).
            indices_to_use = grid_event_indices[1::2]
            print(
                f"  Rule {usage_specifier}: Selecting even-indexed - "
                f"{len(indices_to_use)} of {n_grid_events} grid events."
            )

        elif usage_specifier == 0:  # Use NONE
            print(f"  Rule {usage_specifier}: Selecting no grid events.")
            # indices_to_use remains empty

        else:
            print(f"Warning: Unknown usage_specifier ({usage_specifier}). Selecting no grid events.")
            # indices_to_use remains empty

        # Set the boolean flag for the selected indices in the DataFrame
        if len(indices_to_use) > 0:
            # Use .loc to ensure setting values based on index labels
            df.loc[indices_to_use, output_column_name] = True

        return df

    def pmod_glm1(self, events_df, xFold_local, keep_unused=True):
        """
        Build a Nilearn event table for GLM1 with sine and cosine parametric modulators.
        Selected translation events are mean-centered to match the GridCAT workflow.
        """
        prepared_events_list = []

        selected_grid_events_for_pm = events_df[
            (events_df["trial_type"] == "translation") &
            ~pd.isna(events_df["orientation"]) &
            events_df["use_in_GLM1"]
            ]
        mean_sin_mod = 0.0
        mean_cos_mod = 0.0
        if not selected_grid_events_for_pm.empty:
            grid_angles_rad_selected = np.deg2rad(selected_grid_events_for_pm["orientation"])
            all_sin_mods_selected = np.sin(xFold_local * grid_angles_rad_selected)
            all_cos_mods_selected = np.cos(xFold_local * grid_angles_rad_selected)
            mean_sin_mod = np.mean(all_sin_mods_selected)
            mean_cos_mod = np.mean(all_cos_mods_selected)

        for index, row in events_df.iterrows():
            onset, duration, trial_type = row["onset"], row["duration"], row["trial_type"]
            if pd.isna(onset) or pd.isna(duration) or pd.isna(trial_type):
                continue

            if trial_type == "translation" and not pd.isna(row["orientation"]):
                if row["use_in_GLM1"]:  # Event selected for GLM1
                    grid_angle_rad = np.deg2rad(row["orientation"])
                    sin_mod = np.sin(xFold_local * grid_angle_rad) - mean_sin_mod
                    cos_mod = np.cos(xFold_local * grid_angle_rad) - mean_cos_mod

                    prepared_events_list.append({
                        "onset": onset, "duration": duration,
                        "trial_type": "translation", "modulation": 1.0
                    })
                    prepared_events_list.append({
                        "onset": onset, "duration": duration,
                        "trial_type": "translation_pmodSIN", "modulation": sin_mod
                    })
                    prepared_events_list.append({
                        "onset": onset, "duration": duration,
                        "trial_type": "translation_pmodCOS", "modulation": cos_mod
                    })
                elif keep_unused:  # Event NOT selected for GLM1, but keep_unused is True
                    prepared_events_list.append({
                        "onset": onset, "duration": duration,
                        "trial_type": "translation_unused", "modulation": 1.0
                    })

            elif trial_type == "translation" and pd.isna(row["orientation"]):  # NaN orientation
                if keep_unused:
                    prepared_events_list.append({
                        "onset": onset, "duration": duration,
                        "trial_type": "translation_nan_ori", "modulation": 1.0
                    })

            elif trial_type != "translation":  # Other event types (e.g., 'feedback')
                prepared_events_list.append({
                    "onset": onset, "duration": duration,
                    "trial_type": trial_type, "modulation": 1.0
                })

        if not prepared_events_list:
            # print("pmod_glm1: No events were processed.")
            return pd.DataFrame(columns=["onset", "duration", "trial_type", "modulation"])

        nilearn_events_df = pd.DataFrame(prepared_events_list)
        nilearn_events_df = nilearn_events_df.sort_values(by="onset").reset_index(drop=True)
        # print(f"  pmod_glm1 generated {len(nilearn_events_df)} event entries for Nilearn.")
        return nilearn_events_df[["onset", "duration", "trial_type", "modulation"]]

    def setup_first_level_model(self, t_r, high_pass_period, smoothing_fwhm, slice_time_ref, mask):
        """
        Initialize a Nilearn FirstLevelModel with the shared GridCAT settings.
        """
        model = FirstLevelModel(
            t_r=t_r,
            hrf_model="spm",  # Standard SPM HRF
            verbose=1,
            noise_model='ar1',  # AStandard
            drift_model="cosine",  # Use cosine functions for low-frequency drift
            high_pass=1.0 / high_pass_period,
            standardize=False,
            slice_time_ref = slice_time_ref,
            smoothing_fwhm=smoothing_fwhm,
            mask_img=mask, #or matlab for comparison
            signal_scaling=False
        )
        return model

    def run_glm1_single_run(self, event_table_file, regressor_file, scanFolder_run, run_number,
                            event_usage_specifier, xFold, tr, high_pass_period, smoothing_fwhm, slice_time_ref, mask):
        """
        Read one run, prepare GLM1 events, fit the model, and plot its design matrix.
        Returns the Nilearn event table and fitted FirstLevelModel.
        """
        print(f"Running GLM1 for Run {run_number}")
        # 1. Read data
        event_df = self.read_event_table_to_df(event_table_file)
        add_reg_df = self.read_additional_regressors_to_df(regressor_file)

        # Get functional image files
        func_files = sorted(
            glob.glob(os.path.join(scanFolder_run, "*.nii")) +
            glob.glob(os.path.join(scanFolder_run, "*.nii.gz"))
        )

        print(f"Found {len(func_files)} functional scans for run {run_number}.")
        func_img_4d = concat_imgs(func_files, auto_resample=True)

        #Select events for GLM1
        event_df = self.select_events_for_glm(event_df, event_usage_specifier, "use_in_GLM1")

        #Prepare events for Nilearn using pmod_glm1
        event_df_nilearn = self.pmod_glm1(event_df, xFold)

        # 4. Setup and Fit Model
        # Using mask_path=None for automatic masking within FirstLevelModel
        model = self.setup_first_level_model(tr, high_pass_period, smoothing_fwhm, slice_time_ref, mask)
        print(f"Fitting FirstLevelModel for run {run_number}")

        model.fit(func_img_4d, events=event_df_nilearn, confounds=add_reg_df)

        # Plot design matrix
        design_matrix = model.design_matrices_[0]
        plot_design_matrix(design_matrix)
        plt.suptitle(f"Design Matrix for GLM1 - Run {run_number}")
        plt.show()
        plt.close()

        print(f"GLM1 Fitting complete for run {run_number}.")
        return event_df_nilearn, model

    def betas2ori(self, xFold, beta_sin_vol, beta_cos_vol):
        """
        Convert sine/cosine beta volumes into voxelwise orientation and amplitude maps.
        Orientations are returned in degrees and radians within one x-fold period.
        """
        # Calculate raw orientation per voxel in radians using arctan2(sin, cos)
        # This results in a range of [-pi, pi] for atan2, then scaled by 1/xFold

        orientation_raw_rad = np.arctan2(beta_sin_vol, beta_cos_vol) / xFold

        # Convert raw orientation to degrees first. For xFold=6, this is [-30, 30] degrees.
        orientation_raw_deg = np.rad2deg(orientation_raw_rad)

        # Define the period for the orientation in degrees
        period_deg = 360.0 / xFold

        # Map the raw orientation (in degrees) to the range [0, period_deg)
        # np.mod behaves like MATLAB's mod for a positive divisor, mapping negative inputs appropriately.
        # e.g. np.mod(-10, 60) = 50
        orientation_mapped_deg = np.mod(orientation_raw_deg, period_deg)
        orientation_mapped_rad = np.deg2rad(orientation_mapped_deg)

        # Calculate amplitude per voxel
        amplitude_per_voxel = np.sqrt(np.square(beta_sin_vol) + np.square(beta_cos_vol))

        # Return the mapped orientation in degrees and the amplitude
        return orientation_mapped_deg, orientation_mapped_rad, amplitude_per_voxel


    def calculate_betas(self, run_indices_glm1_success, glm1_models, xFold, grid_ori_output_dir):
        """
        Compute and save GLM1 orientation and amplitude maps for each successful run.
        Also saves averaged maps when more than one run is available.
        """
        # Calculate and save results for each successful run
        beta_sin_vols = {}
        beta_cos_vols = {}
        ref_img_for_saving = None  # Use one image as spatial reference

        for run_idx in run_indices_glm1_success:
            print(f"Calculating orientation/amplitude for GLM1 run {run_idx}")
            model = glm1_models[run_idx]

            # Compute contrast maps (effect size = beta estimates here)
            beta_sin_img = model.compute_contrast('translation_pmodSIN', output_type='effect_size')
            beta_cos_img = model.compute_contrast('translation_pmodCOS', output_type='effect_size')

            if ref_img_for_saving is None:
                ref_img_for_saving = beta_sin_img  # Use first available image as reference

            # Get data volumes
            beta_sin_vol = beta_sin_img.get_fdata()
            beta_cos_vol = beta_cos_img.get_fdata()
            beta_sin_vols[run_idx] = beta_sin_vol
            beta_cos_vols[run_idx] = beta_cos_vol

            # Set empty beta values to NaN like Matlab.
            beta_sin_vol[beta_sin_vol == 0.0] = np.nan
            beta_cos_vol[beta_cos_vol == 0.0] = np.nan

            # Calculate orientation and amplitude
            ori_deg, ori_rad, amp = self.betas2ori(xFold, beta_sin_vol, beta_cos_vol)

            # Save run-specific results as NIfTI images
            ori_deg_img_run = new_img_like(ref_img_for_saving, ori_deg)
            ori_rad_img_run = new_img_like(ref_img_for_saving, ori_rad)
            amp_img_run = new_img_like(ref_img_for_saving, amp)

            nib.save(
                ori_deg_img_run,
                os.path.join(grid_ori_output_dir, f'voxelwiseGridOri_translation_run{run_idx}_deg.nii')
            )
            nib.save(
                ori_rad_img_run,
                os.path.join(grid_ori_output_dir, f'voxelwiseGridOri_translation_run{run_idx}_rad.nii')
            )
            nib.save(
                amp_img_run,
                os.path.join(grid_ori_output_dir, f'voxelwiseAmplitude_translation_run{run_idx}.nii')
            )
            print(f"  Saved orientation and amplitude maps for run {run_idx}.")

        if len(run_indices_glm1_success) > 1:
            print("Calculating average orientation/amplitude for GLM1")

            # Ensure all vols have the same shape before averaging
            first_shape = beta_sin_vols[run_indices_glm1_success[0]].shape

            beta_sin_avg = np.mean(np.stack([beta_sin_vols[r] for r in run_indices_glm1_success]), axis=0)
            beta_cos_avg = np.mean(np.stack([beta_cos_vols[r] for r in run_indices_glm1_success]), axis=0)

            ori_avg_deg, _, amp_avg = self.betas2ori(xFold, beta_sin_avg, beta_cos_avg)

            ori_avg_img = new_img_like(ref_img_for_saving, ori_avg_deg)
            amp_avg_img = new_img_like(ref_img_for_saving, amp_avg)

            nib.save(
                ori_avg_img,
                os.path.join(grid_ori_output_dir, 'voxelwiseGridOri_translation_allRunsAvg_deg.nii')
            )
            nib.save(
                amp_avg_img,
                os.path.join(grid_ori_output_dir, 'voxelwiseAmplitude_translation_allRunsAvg.nii')
            )
            print("Saved averaged orientation and amplitude maps.")
        else:
            print("Only one run available; cannot average.")

    def calculate_mean_grid_orientation(self, glm1_dir, roi_mask_path, xFold, use_weighting, average_across_runs,
                                        run_indices, grid_event_name="translation"):
        """
        Calculate mean grid orientation within an ROI from GLM1 orientation maps.
        Can return a single averaged orientation or per-run orientations.
        """
        grid_ori_subdir = os.path.join(glm1_dir, "GridOrientation")
        roi_mask_path = os.path.join(self.base_path, "ROI_masks/ROImask_entorhinalCortex_RH.nii")
        print(f" Calculating Mean Grid Orientation ({'Avg Across Runs' if average_across_runs else 'Per Run'})")
        print(f"  GLM1 Dir: {grid_ori_subdir}")
        print(f"  ROI Mask: {roi_mask_path}")
        print(f"  Weighting: {use_weighting}, AverageRuns Flag: {average_across_runs}")

        try:
            roi_img = load_img(roi_mask_path)
            print(f"  Loaded ROI mask.")
        except Exception as e:
            print(f"  ERROR: Could not load ROI mask '{roi_mask_path}': {e}")
            print("debug")
            return None # Cannot proceed without ROI mask

        if average_across_runs:
            avg_ori_file = os.path.join(grid_ori_subdir, f'voxelwiseGridOri_{grid_event_name}_allRunsAvg_deg.nii')
            avg_amp_file = os.path.join(grid_ori_subdir, f'voxelwiseAmplitude_{grid_event_name}_allRunsAvg.nii')

            # Load the pre-averaged orientation map
            avg_ori_img_load = load_img(avg_ori_file)
            masked_avg_ori_deg = apply_mask(avg_ori_img_load, roi_img)

            # Initialize weights as None (will be set if weighting is used)
            weights = None
            if use_weighting:
                # Load the pre-averaged amplitude map only if weighting
                avg_amp_img_load = load_img(avg_amp_file)
                masked_avg_amp = apply_mask(avg_amp_img_load, roi_img)
                weights = masked_avg_amp  # Use the masked amplitude as weights

            # Convert degrees to radians for calculation
            ori_rad = np.deg2rad(masked_avg_ori_deg)

            # Create mask for valid (non-NaN) orientation values
            valid_mask = ~np.isnan(ori_rad)

            if use_weighting:
                # Also exclude NaN weights and optionally non-positive weights
                valid_mask &= ~np.isnan(weights)  # Exclude NaN weights
                weights_clean = weights[valid_mask]
            else:
                weights_clean = None  # No weighting used

            # Calculate components only for valid voxels
            ori_rad_clean = ori_rad[valid_mask]

            # SMap orientations to full circle using xFold
            # Then convert each angle to a unit vector on the circle
            sin_comp = np.sin(xFold * ori_rad_clean)  # Y-component of unit vector
            cos_comp = np.cos(xFold * ori_rad_clean)  # X-component of unit vector

            if use_weighting:
                # Instead of: mean(angles * weights)
                # We do: mean(unit_vectors * weights)

                total_weight = np.sum(weights_clean)
                if total_weight > 0:
                    # Calculate weighted mean of the vector components
                    mean_sin = np.sum(sin_comp * weights_clean) / total_weight  # Mean Y-component
                    mean_cos = np.sum(cos_comp * weights_clean) / total_weight  # Mean X-component
                    raw_mean_ori_rad = np.arctan2(mean_sin, mean_cos) / xFold
                    period_rad = (2 * np.pi) / xFold
                    mean_ori_rad_result = np.mod(raw_mean_ori_rad, period_rad)

                    print(f" Calculated ROI Mean Orientation (Avg Runs): {np.rad2deg(mean_ori_rad_result):.3f} deg")

                    return mean_ori_rad_result

                else:
                    print("  Warning: Total weight of valid voxels is zero.")
                    return np.nan
            else:
                # Unweighted average of components
                print("Using Unweighted")
                mean_sin = np.mean(sin_comp)
                mean_cos = np.mean(cos_comp)

                raw_mean_ori_rad = np.arctan2(mean_sin, mean_cos) / xFold

                # Remap the raw mean orientation to radians
                period_rad = (2 * np.pi) / xFold
                mean_ori_rad_mapped = np.mod(raw_mean_ori_rad, period_rad)

                mean_ori_rad_result = mean_ori_rad_mapped # Use the mapped result

                print(f"  Calculated ROI Mean Orientation (Avg Runs): {np.rad2deg(mean_ori_rad_result):.3f} deg")
                return mean_ori_rad_result

        else:

            results = {}

            # --- Per Run Logic
            print("  Calculating mean orientation per run...")
            results = {}
            # Determine which runs have data available
            available_runs = []
            for run_idx in run_indices:
                ori_file = os.path.join(grid_ori_subdir, f'voxelwiseGridOri_{grid_event_name}_run{run_idx}_deg.nii')
                amp_file = os.path.join(grid_ori_subdir, f'voxelwiseAmplitude_{grid_event_name}_run{run_idx}.nii')
                # Load the pre-averaged orientation map
                avg_ori_img_load = load_img(ori_file)
                masked_avg_ori_deg = apply_mask(avg_ori_img_load, roi_img)

                # Initialize weights as None (will be set if weighting is used)
                weights = None
                if use_weighting:
                    # Load the pre-averaged amplitude map only if weighting
                    avg_amp_img_load = load_img(amp_file)
                    masked_avg_amp = apply_mask(avg_amp_img_load, roi_img)
                    weights = masked_avg_amp  # Use the masked amplitude as weights

                ori_rad = np.deg2rad(masked_avg_ori_deg)

                valid_mask = ~np.isnan(ori_rad)

                if use_weighting:
                    valid_mask &= ~np.isnan(weights)  # Exclude NaN weights
                    weights_clean = weights[valid_mask]
                else:
                    weights_clean = None  # No weighting used

                ori_rad_clean = ori_rad[valid_mask]

                sin_comp = np.sin(xFold * ori_rad_clean)
                cos_comp = np.cos(xFold * ori_rad_clean)

                if use_weighting:

                    total_weight = np.sum(weights_clean)
                    if total_weight > 0:
                        mean_sin = np.sum(sin_comp * weights_clean) / total_weight  # Mean Y-component
                        mean_cos = np.sum(cos_comp * weights_clean) / total_weight  # Mean X-component

                        raw_mean_ori_rad = np.arctan2(mean_sin, mean_cos) / xFold

                        period_rad = (2 * np.pi) / xFold
                        mean_ori_rad_result = np.mod(raw_mean_ori_rad, period_rad)

                        print(
                            f" Calculated ROI Mean Orientation (Avg {run_idx}): "
                            f"{np.rad2deg(mean_ori_rad_result):.3f} deg"
                        )

                        run_result = np.rad2deg(mean_ori_rad_result)

                        results[f"{run_idx}"] = run_result

                else:

                    mean_sin = np.sum(sin_comp)  # Mean Y-component
                    mean_cos = np.sum(cos_comp)

                    raw_mean_ori_rad = np.arctan2(mean_sin, mean_cos) / xFold

                    period_rad = (2 * np.pi) / xFold
                    mean_ori_rad_result = np.mod(raw_mean_ori_rad, period_rad)

                    print("average Ori")
                    print(
                        f" Calculated ROI Mean Orientation (Avg {run_idx}): "
                        f"{np.rad2deg(mean_ori_rad_result):.3f} deg"
                    )

                    run_result = np.rad2deg(mean_ori_rad_result)

                    results[f"{run_idx}"] = run_result


            return results

    def prepare_glm2_events(self, base_event_df, mean_ori_rad, xFold, method, keep_unused_events, run):
        """
        Prepare a Nilearn event table for GLM2 using alignment to the mean orientation.
        Supports parametric modulation and aligned/misaligned event coding.
        """
        print(f"Preparing Events for GLM2 using method: {method}")
        prepared_events_list = []
        event_suffix = "_test"

        for index, row in base_event_df.iterrows():
            onset = row["onset"]
            duration = row["duration"]
            trial_type = row["trial_type"]

            if pd.isna(onset) or pd.isna(duration) or pd.isna(trial_type):
                print(f"Warning: Skipping event at index {index} due to NaN onset/duration/trial_type.")
                continue

            if trial_type == "translation" and not pd.isna(row["orientation"]):
                if row["use_in_GLM2"]:
                    ori_rad = np.deg2rad(row["orientation"])
                    alignment = np.cos(xFold * (ori_rad - mean_ori_rad))

                    if method == 'pmod':
                        prepared_events_list.append({
                            "onset": onset, "duration": duration,
                            "trial_type": f"{trial_type}{event_suffix}",
                            "modulation": 1.0
                        })
                        modulation_sign = -1 if run == 1 else 1
                        prepared_events_list.append({
                            "onset": onset, "duration": duration,
                            "trial_type": f"{trial_type}{event_suffix}_pmodAlignment",
                            "modulation": alignment * modulation_sign
                        })
                    elif method == 'aligned_misaligned':
                        label = "aligned" if alignment >= 0 else "misaligned"
                        prepared_events_list.append({
                            "onset": onset, "duration": duration,
                            "trial_type": f"{trial_type}{event_suffix}_{label}",
                            "modulation": 1.0
                        })

                elif keep_unused_events:
                    prepared_events_list.append({
                        "onset": onset, "duration": duration,
                        "trial_type": f"{trial_type}{event_suffix}_unused",
                        "modulation": 1.0
                    })

            elif trial_type != "translation":
                prepared_events_list.append({
                    "onset": onset, "duration": duration,
                    "trial_type": f"{trial_type}{event_suffix}",
                    "modulation": 1.0
                })

        nilearn_events_df = pd.DataFrame(prepared_events_list)
        nilearn_events_df = nilearn_events_df.sort_values(by="onset").reset_index(drop=True)
        print(f"prepare_glm2_events generated {len(nilearn_events_df)} event entries for Nilearn.")
        return nilearn_events_df[["onset", "duration", "trial_type", "modulation"]]

    def compute_glm2_contrasts(self, model_1, model_2, model_a, method, grid_event_name):
        """
        Compute GLM2 contrasts for run 1, run 2, and the combined model.
        """
        print("Computing GLM2 Contrasts...")
        event_suffix = "_test"
        base_contrast_name = f"{grid_event_name}{event_suffix}"
        available_regressors = list(model_1.design_matrices_[0].columns)

        if method == 'pmod':
            contrast_definitions = f"{base_contrast_name}_pmodAlignment"
        elif method == 'aligned_misaligned':
            aligned_reg = f"{grid_event_name}{event_suffix}_aligned"
            misaligned_reg = f"{grid_event_name}{event_suffix}_misaligned"
            weights = np.zeros(len(available_regressors))
            weights[available_regressors.index(aligned_reg)] = 1
            weights[available_regressors.index(misaligned_reg)] = -1
            contrast_definitions = weights
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'pmod' or 'aligned_misaligned'.")

        computed_contrasts = {
            f"{method}_Run1": model_1.compute_contrast(
                contrast_definitions, output_type='effect_size', stat_type="t"
            ),
            f"{method}_Run2": model_2.compute_contrast(
                contrast_definitions, output_type='effect_size', stat_type="t"
            ),
            f"{method}_RunAverage": model_a.compute_contrast(
                contrast_definitions, output_type='effect_size', stat_type="t"
            ),
        }
        print(f"Successfully computed {len(computed_contrasts)} GLM2 contrasts.")
        return computed_contrasts

    def plot_roi_orientation_histogram(self, orientations_deg, mean_orientation_deg, xFold, roi_name, image_fname):
        """
        Display a polar histogram of voxelwise grid orientations for one ROI.
        """
        valid_orientations = orientations_deg[~np.isnan(orientations_deg)]
        if valid_orientations.size == 0:
            print(f"Skipping plot for {roi_name}: No valid orientation voxels found.")
            return

        plot_angles_rad = np.deg2rad(valid_orientations * xFold)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        n_bins = 24
        hist, bin_edges = np.histogram(plot_angles_rad, bins=n_bins, range=(0, 2 * np.pi))
        angles = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = (2 * np.pi) / n_bins

        ax.bar(angles, hist, width=width, alpha=0.7, edgecolor='k')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        tick_angles_deg_folded = np.arange(0, 360, 60)
        tick_labels_unfolded = [f'{int(angle / xFold)}°' for angle in tick_angles_deg_folded]
        ax.set_xticks(np.deg2rad(tick_angles_deg_folded))
        ax.set_xticklabels(tick_labels_unfolded)

        if not np.isnan(mean_orientation_deg):
            mean_plot_angle_rad = np.deg2rad(mean_orientation_deg * xFold)
            ax.annotate('', xy=(mean_plot_angle_rad, ax.get_ylim()[1]), xytext=(0, 0),
                        arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=10,
                                        edgecolor='black', linewidth=0.5))

        ax.set_title(
            f"Grid Orientation Distribution in {os.path.basename(roi_name)}\n"
            f"(from {os.path.basename(image_fname)})",
            fontsize=16, pad=20
        )
        plt.tight_layout()
        plt.show()

    def calculate_and_export_metrics(self, roi_mask_paths, glm1_dir, glm2_contrasts, output_file, xFold,
                                     grid_event_name):
        """
        Calculate GridCAT-style metrics and export them to a semicolon-separated text file.
        Returns the formatted text and the GLM2 contrast objects.
        """
        print(f"Calculating and Exporting Grid Metrics to: {output_file}")
        all_metrics_lines = []
        sep = ";"

        grid_ori_subdir = os.path.join(glm1_dir, "GridOrientation")
        print(f"  Using GLM1 orientation/amplitude files from: {grid_ori_subdir}")

        all_files_in_subdir = os.listdir(grid_ori_subdir)
        glm1_ori_files = sorted([
            f for f in all_files_in_subdir
            if f.startswith(f'voxelwiseGridOri_{grid_event_name}') and f.endswith('.nii')
            and '_run' in f and 'allRunsAvg' not in f
        ])
        glm1_amp_files = sorted([
            f for f in all_files_in_subdir
            if f.startswith(f'voxelwiseAmplitude_{grid_event_name}') and f.endswith('.nii')
            and '_run' in f and 'allRunsAvg' not in f
        ])

        avg_ori_fname = f'voxelwiseGridOri_{grid_event_name}_allRunsAvg_deg.nii'
        avg_amp_fname = f'voxelwiseAmplitude_{grid_event_name}_allRunsAvg.nii'
        avg_ori_file_path = (
            os.path.join(grid_ori_subdir, avg_ori_fname)
            if avg_ori_fname in all_files_in_subdir else None
        )
        avg_amp_file_path = (
            os.path.join(grid_ori_subdir, avg_amp_fname)
            if avg_amp_fname in all_files_in_subdir else None
        )

        inferred_run_indices = []
        if glm1_ori_files:
            runs = [int(f.split('_run')[1].split('_')[0]) for f in glm1_ori_files]
            inferred_run_indices = sorted(set(runs))
            print(f"  Inferred run indices: {inferred_run_indices}")

        # Metric 1: Magnitude of grid code response within ROI (GLM2)
        print("\nMetric 1: Grid Code Response Magnitude (GLM2)")
        header1 = [
            "GRID METRIC", "X-FOLD SYMMETRY", "ROI", "CONTRAST NUMBER",
            "CONTRAST NAME", "MEAN CON-VALUE WITHIN ROI", "VOXELS WITHIN ROI", "NaN VOXELS WITHIN ROI"
        ]
        all_metrics_lines.append(sep.join(header1))

        for roi_path in roi_mask_paths:
            roi_name = os.path.basename(roi_path)
            roi_img = load_img(roi_path)
            for con_idx, (con_name, con_map) in enumerate(glm2_contrasts.items()):
                masked_con_data = apply_mask(con_map, roi_img)
                mean_val = np.nanmean(masked_con_data)
                n_vox = len(masked_con_data)
                n_nan = np.sum(np.isnan(masked_con_data))
                row = [
                    "Magnitude of grid code response within ROI",
                    str(xFold), roi_name, str(con_idx + 1), con_name,
                    f"{mean_val:.3f}" if not pd.isna(mean_val) else "NaN",
                    str(n_vox), str(n_nan)
                ]
                all_metrics_lines.append(sep.join(row))
        all_metrics_lines.append("")

        # Metric 2: Between-voxel grid orientation coherence (GLM1)
        print("\nMetric 2: Between-Voxel Coherence (GLM1)")
        header2 = ["GRID METRIC", "X-FOLD SYMMETRY", "ROI", "VOXEL-WISE GRID ORI IMAGE",
                   "RAYLEIGH z", "RAYLEIGH p", "VOXELS WITHIN ROI", "NaN VOXELS WITHIN ROI"]
        all_metrics_lines.append(sep.join(header2))

        all_ori_files = [f for f in glm1_ori_files if f.endswith("deg.nii")]
        if avg_ori_file_path and avg_ori_file_path.endswith("deg.nii"):
            all_ori_files.append(os.path.basename(avg_ori_file_path))

        for roi_path in roi_mask_paths:
            roi_name = os.path.basename(roi_path)
            roi_img = load_img(roi_path)
            for ori_fname in all_ori_files:
                ori_img_load = load_img(os.path.join(grid_ori_subdir, ori_fname))
                masked_ori_deg = apply_mask(ori_img_load, roi_img)
                n_vox = len(masked_ori_deg)
                n_nan = np.sum(np.isnan(masked_ori_deg))
                valid_ori_deg = masked_ori_deg[~np.isnan(masked_ori_deg)]
                ori_rad_folded = np.deg2rad(valid_ori_deg) * xFold
                rayleigh_p, rayleigh_z = pycircstat.tests.rayleigh(ori_rad_folded)
                row = [
                    "Between-voxel grid orientation coherence within ROI",
                    str(xFold), roi_name, ori_fname,
                    f"{rayleigh_z:.3f}", f"{rayleigh_p:.5f}",
                    str(n_vox), str(n_nan)
                ]
                all_metrics_lines.append(sep.join(row))
        all_metrics_lines.append("")

        # Metric 3: Within-voxel grid orientation coherence (GLM1)
        print("\nMetric 3: Within-Voxel Coherence (GLM1)")
        header3 = ["GRID METRIC", "X-FOLD SYMMETRY", "ROI", "VOXEL-WISE GRID ORI IMAGE 1",
                   "VOXEL-WISE GRID ORI IMAGE 2", "% STABLE VOXELS WITHIN ROI",
                   "IMAGE 1 VOXELS WITHIN ROI", "IMAGE 2 VOXELS WITHIN ROI",
                   "IMAGE 1 NaN VOXELS WITHIN ROI", "IMAGE 2 NaN VOXELS WITHIN ROI"]
        all_metrics_lines.append(sep.join(header3))

        stability_threshold_rad_folded = np.pi / 2

        for roi_path in roi_mask_paths:
            roi_name = os.path.basename(roi_path)
            roi_img = load_img(roi_path)
            masked_data_cache = {}
            for i, fname in enumerate(all_ori_files):
                try:
                    img_load = load_img(os.path.join(grid_ori_subdir, fname))
                    masked_ori_deg = apply_mask(img_load, roi_img)
                    masked_data_cache[i] = {
                        'fname': fname, 'data': masked_ori_deg if masked_ori_deg.size > 0 else None,
                        'n_vox': len(masked_ori_deg), 'n_nan': np.sum(np.isnan(masked_ori_deg))
                    }
                except Exception as e_load:
                    print(f"    Error loading {fname}: {e_load}")
                    masked_data_cache[i] = {'fname': fname, 'data': None, 'n_vox': 0, 'n_nan': 0}

            for i in range(len(all_ori_files)):
                for j in range(i + 1, len(all_ori_files)):
                    d1, d2 = masked_data_cache[i], masked_data_cache[j]
                    if d1['data'] is None or d2['data'] is None:
                        continue
                    valid_both = ~np.isnan(d1['data']) & ~np.isnan(d2['data'])
                    n_valid_both = np.sum(valid_both)
                    if n_valid_both == 0:
                        continue
                    ori1_rad = np.deg2rad(d1['data'][valid_both])
                    ori2_rad = np.deg2rad(d2['data'][valid_both])
                    diff_folded = np.angle(
                        np.exp(1j * (ori1_rad * xFold)) / np.exp(1j * (ori2_rad * xFold))
                    )
                    percent_stable = (
                        np.sum(np.abs(diff_folded) <= stability_threshold_rad_folded) / n_valid_both
                    ) * 100
                    row = [
                        "Within-voxel grid orientation coherence within ROI",
                        str(xFold), roi_name, d1['fname'], d2['fname'],
                        f"{percent_stable:.2f}", str(d1['n_vox']), str(d2['n_vox']),
                        str(d1['n_nan']), str(d2['n_nan'])
                    ]
                    all_metrics_lines.append(sep.join(row))
        all_metrics_lines.append("")

        # Metric 4: Mean grid orientation within ROI (GLM1)
        print("\nMetric 4: Mean Grid Orientation (calculated from GLM1)")
        header4 = ["GRID METRIC", "X-FOLD SYMMETRY", "ROI", "GRID EVENT", "RUN",
                   "MEAN GRID ORI IN DEGREES (VOXELS WEIGHTED)", "MEAN GRID ORI IN DEGREES (VOXELS NOT WEIGHTED)"]
        all_metrics_lines.append(sep.join(header4))

        for roi_path in roi_mask_paths:
            roi_name = os.path.basename(roi_path)
            if inferred_run_indices:
                mean_ori_w = self.calculate_mean_grid_orientation(
                    glm1_dir=glm1_dir, roi_mask_path=roi_path, xFold=xFold,
                    use_weighting=True, average_across_runs=False,
                    run_indices=inferred_run_indices, grid_event_name=grid_event_name
                )
                mean_ori_nw = self.calculate_mean_grid_orientation(
                    glm1_dir=glm1_dir, roi_mask_path=roi_path, xFold=xFold,
                    use_weighting=False, average_across_runs=False,
                    run_indices=inferred_run_indices, grid_event_name=grid_event_name
                )
                if isinstance(mean_ori_w, dict) and isinstance(mean_ori_nw, dict):
                    for run_idx in inferred_run_indices:
                        ori_w = mean_ori_w.get(str(run_idx), np.nan)
                        ori_nw = mean_ori_nw.get(str(run_idx), np.nan)
                        row = [
                            "Mean grid orientation within ROI", str(xFold), roi_name,
                            grid_event_name, str(run_idx),
                            f"{ori_w:.3f}" if not pd.isna(ori_w) else "NaN",
                            f"{ori_nw:.3f}" if not pd.isna(ori_nw) else "NaN"
                        ]
                        all_metrics_lines.append(sep.join(row))

            try:
                mean_ori_w_avg = self.calculate_mean_grid_orientation(
                    glm1_dir=glm1_dir, roi_mask_path=roi_path, xFold=xFold,
                    use_weighting=True, average_across_runs=True,
                    run_indices=inferred_run_indices, grid_event_name=grid_event_name
                )
                mean_ori_nw_avg = self.calculate_mean_grid_orientation(
                    glm1_dir=glm1_dir, roi_mask_path=roi_path, xFold=xFold,
                    use_weighting=False, average_across_runs=True,
                    run_indices=inferred_run_indices, grid_event_name=grid_event_name
                )
                row_avg = [
                    "Mean grid orientation within ROI", str(xFold), roi_name,
                    grid_event_name, "averaged across runs",
                    (
                        f"{np.rad2deg(mean_ori_w_avg):.3f}"
                        if mean_ori_w_avg is not None and not pd.isna(mean_ori_w_avg) else "NaN"
                    ),
                    (
                        f"{np.rad2deg(mean_ori_nw_avg):.3f}"
                        if mean_ori_nw_avg is not None and not pd.isna(mean_ori_nw_avg) else "NaN"
                    )
                ]
                all_metrics_lines.append(sep.join(row_avg))

                if avg_ori_file_path and mean_ori_w_avg is not None:
                    avg_ori_img = load_img(avg_ori_file_path)
                    masked_avg_ori_deg = apply_mask(avg_ori_img, load_img(roi_path))
                    self.plot_roi_orientation_histogram(
                        orientations_deg=masked_avg_ori_deg,
                        mean_orientation_deg=np.rad2deg(mean_ori_w_avg),
                        xFold=xFold, roi_name=roi_name,
                        image_fname=os.path.basename(avg_ori_file_path)
                    )
            except Exception as e:
                print(f"  Error calculating averaged mean orientations for '{roi_name}': {e}")
                all_metrics_lines.append(sep.join([
                    "Mean grid orientation within ROI", str(xFold), roi_name,
                    grid_event_name, "averaged across runs", "CalcError", "CalcError"
                ]))
        all_metrics_lines.append("")

        with open(output_file, 'w') as f:
            for line in all_metrics_lines:
                f.write(line + '\n')
        print(f"Successfully wrote metrics to {output_file}")

        return "\n".join(all_metrics_lines), glm2_contrasts
