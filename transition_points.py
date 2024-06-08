import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import statsmodels.api as sm
import itertools
from tqdm import tqdm

def cusum_change_points(y, threshold_factor=1, smoothing_sigma=2, min_distance=10):
    y_smoothed = gaussian_filter1d(y, sigma=smoothing_sigma)
    mean_y = np.mean(y_smoothed)
    cumsum = np.cumsum(y_smoothed - mean_y)
    dynamic_threshold = threshold_factor * np.std(cumsum)

    change_points = []
    last_cp = -min_distance
    for i in range(len(cumsum)):
        if abs(cumsum[i]) > dynamic_threshold and i - last_cp >= min_distance:
            if i > 0:
                segment1, segment2 = y[:i], y[i:]
                if len(segment1) >= 3 and len(segment2) >= 3:
                    p_value_shapiro1, p_value_shapiro2 = shapiro(segment1).pvalue, shapiro(segment2).pvalue
                    alpha = 0.05
                    if p_value_shapiro1 > alpha and p_value_shapiro2 > alpha:
                        p_value = ttest_ind(segment1, segment2).pvalue
                    else:
                        p_value = mannwhitneyu(segment1, segment2).pvalue

                    if p_value < alpha:
                        change_points.append(i)
                        last_cp = i
    return change_points

def calculate_frac(segment, x):
    std_dev = x[segment.index].std()
    max_frac = 0.999
    min_frac = 0.100
    frac = 1 / (1 + std_dev)
    frac = min(max_frac, max(min_frac, frac))
    return frac

def is_valid_segment(segment, min_size=4):
    return len(segment) >= min_size


def dynamic_threshold_three_cluster_reduction(lst, fraction_of_std_dev=0.5):
    if len(lst) < 2:
        return lst

    std_dev = np.std(lst)
    dynamic_range_threshold = std_dev * fraction_of_std_dev
    clusters = []
    current_cluster = [lst[0]]

    for value in lst[1:]:
        if abs(value - current_cluster[0]) <= dynamic_range_threshold:
            current_cluster.append(value)
        else:
            clusters.append(np.median(current_cluster))
            current_cluster = [value]
    clusters.append(np.median(current_cluster))
    while len(clusters) > 3:
        closest_distance = float('inf')
        merge_index = 0

        for i in range(len(clusters) - 1):
            distance = abs(clusters[i] - clusters[i + 1])
            if distance < closest_distance:
                closest_distance = distance
                merge_index = i
        merged_cluster = np.median([clusters[merge_index], clusters[merge_index + 1]])
        clusters.pop(merge_index + 1)
        clusters[merge_index] = merged_cluster

    return clusters
    
def find_transition_point(x, y, slope_thresholds=[0.05], matching_thresholds=[0.001], peak_bottom_thresholds=[0.9], significance_thresholds=[0.001], user_defined_s=[0.05], user_defined_k=[3], handle_duplicates='median'):
    best_result = None
    all_hyperparameter_transition_points = []
    all_transition_points = []
    x_smoothed_all = []
    y_sv_smoothed_all = []
    x_spline_smooth_all = []
    y_spline_smooth_all = []
    hyperparameter_combinations = list(itertools.product(slope_thresholds, matching_thresholds, peak_bottom_thresholds, significance_thresholds, user_defined_s, user_defined_k))
    
    for combination in tqdm(hyperparameter_combinations, desc="TPE running"):
        slope_threshold, matching_threshold, peak_bottom_threshold, significance_threshold, s, k = combination

        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError("Initial data contains NaN values")

        # num_segments_options = list(np.arange(1, 20, 2))
        num_segments_options = list(np.arange(1, 4, 2))

        for num_segments in num_segments_options:
            quantiles = np.linspace(0, 1, num_segments + 1)
            segments = pd.qcut(x, quantiles, duplicates='drop')

            smoothed_segments = []
            for segment_label, segment in segments.groupby(segments):
                if len(segment) >= 4:
                    std_dev = x[segment.index].std()
                    frac = 1 / (1 + std_dev)
                    lowess = sm.nonparametric.lowess(y[segment.index], x[segment.index], frac=frac)
                    if np.isnan(lowess).any():
                        raise ValueError("LOWESS is producing NaN values")
                    smoothed_segments.append(lowess)

            combined_smoothed = np.vstack(smoothed_segments)
            x_smoothed, y_sv_smoothed = combined_smoothed[:, 0], combined_smoothed[:, 1]

            if handle_duplicates in ['mean', 'median']:
                df = pd.DataFrame({'x': x_smoothed, 'y': y_sv_smoothed})
                df = df.groupby('x').agg(handle_duplicates).reset_index()
                x_smoothed, y_sv_smoothed = df['x'].values, df['y'].values

            x_smoothed = np.nan_to_num(x_smoothed, nan=np.nanmean(x_smoothed))
            y_sv_smoothed = np.nan_to_num(y_sv_smoothed, nan=np.nanmean(y_sv_smoothed))

            weights = np.abs(y_sv_smoothed) / np.max(np.abs(y_sv_smoothed))
            weights[weights < 0.1] = 0.1

            if len(x_smoothed) > 1 and len(y_sv_smoothed) > 1:
                spline = UnivariateSpline(x_smoothed, y_sv_smoothed, w=weights, k=k)
                spline.set_smoothing_factor(s=s)

                try:
                    x_min, x_max = x.min(), x.max()
                    min_range = 1e-6
                    if (x_max - x_min) < min_range:
                        mid_point = (x_max + x_min) / 2
                        x_min, x_max = mid_point - min_range / 2, mid_point + min_range / 2
                    x_spline_smooth = np.linspace(x_min, x_max, 1000)
                    y_spline_smooth = spline(x_spline_smooth)
                except Exception as e:
                    print("Error during spline evaluation:", e)
                    continue

                x_smoothed_all.append(x_smoothed)
                y_sv_smoothed_all.append(y_sv_smoothed)
                x_spline_smooth_all.append(x_spline_smooth)
                y_spline_smooth_all.append(y_spline_smooth)

                spline_first_derivative = spline.derivative(n=1)
                spline_second_derivative = spline.derivative(n=2)

                spline_transition_points = []
                for i in range(1, len(x_spline_smooth)):
                    if np.sign(y_spline_smooth[i - 1]) != np.sign(y_spline_smooth[i]):
                        slope_at_crossing = spline_first_derivative(x_spline_smooth[i])
                        if abs(slope_at_crossing) > slope_threshold:
                            spline_transition_points.append(x_spline_smooth[i])

                change_points = cusum_change_points(y)
                for cp in change_points:
                    try:
                        if cp < len(y):
                            x_value_at_cp = x_smoothed[cp]

                            if spline_transition_points:
                                closest_spline_point = min(spline_transition_points, key=lambda point: abs(point - x_value_at_cp))
                                
                                if abs(closest_spline_point - x_value_at_cp) < matching_threshold:
                                    all_transition_points.append(closest_spline_point)
                            # else:
                            #     print("No spline change point to the feature value found. Skipped.")
                        else:
                            print(f"Change point {cp} is out of range.")
                    except Exception as e:
                        continue
                        # print(f"CUSUM change point {cp}: {e}")

                second_derivative = spline_second_derivative(x_spline_smooth)
                peaks = argrelextrema(second_derivative, np.greater)[0]
                bottoms = argrelextrema(second_derivative, np.less)[0]

                if not isinstance(all_transition_points, list):
                    all_transition_points = all_transition_points.tolist()

                for idx in np.concatenate((peaks, bottoms)):
                    if abs(y_spline_smooth[idx]) < peak_bottom_threshold:
                        if abs(second_derivative[idx]) > significance_threshold:
                            potential_transition_point = x_spline_smooth[idx]
                            all_transition_points.append(potential_transition_point)

            all_transition_points = np.unique(all_transition_points)
            all_hyperparameter_transition_points.extend(all_transition_points)


    max_length = max(len(a) for a in x_smoothed_all)
    padded_arrays = [np.pad(a, (0, max_length - len(a)), constant_values=np.nan) for a in x_smoothed_all]
    x_smoothed_avg = np.nanmean(np.array(padded_arrays), axis=0)

    max_length = max(len(a) for a in y_sv_smoothed_all)
    padded_arrays = [np.pad(a, (0, max_length - len(a)), constant_values=np.nan) for a in y_sv_smoothed_all]
    y_sv_smoothed_avg = np.nanmean(np.array(padded_arrays), axis=0)

    x_spline_smooth_avg = np.mean(np.array(x_spline_smooth_all), axis=0)
    y_spline_smooth_avg = np.mean(np.array(y_spline_smooth_all), axis=0)

    all_hyperparameter_transition_points = np.unique(all_hyperparameter_transition_points)
    sorted_transition_points = sorted(all_hyperparameter_transition_points)
    outlier_threshold = 1 
    filtered_sorted_transition_points = [point for point in sorted_transition_points if abs(point - np.mean(sorted_transition_points)) <= outlier_threshold * np.std(sorted_transition_points)]

    if np.isnan(filtered_sorted_transition_points).any():
        best_result = (x_smoothed_avg, y_sv_smoothed_avg, x_spline_smooth_avg, y_spline_smooth_avg, None, None)
    else:    
        filtered_transition_points = dynamic_threshold_three_cluster_reduction(filtered_sorted_transition_points)
        best_result = (x_smoothed_avg, y_sv_smoothed_avg, x_spline_smooth_avg, y_spline_smooth_avg, len(filtered_transition_points) > 0, [np.mean(filtered_transition_points)]) # mean of the three cluster points (default)
        # best_result = (x_smoothed_avg, y_sv_smoothed_avg, x_spline_smooth_avg, y_spline_smooth_avg, len(filtered_transition_points) > 0, filtered_transition_points) # all three points
    return best_result