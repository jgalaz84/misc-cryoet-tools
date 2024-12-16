#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2024; last modification: Dec/13/2024

import h5py
import numpy as np
import argparse
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import matplotlib.pyplot as plt
from sklearn.utils import resample
from molecular_stats import calculate_metrics_for_images, is_normal, calculate_cohens_d, calculate_rank_biserial, create_output_directory
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor

def setup_logger(log_file):
    """Set up a logger to record script execution details."""
    logger = logging.getLogger('molecular_stats_bootstrap')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def list_all_hdf5_paths(file_path):
    """List all paths in an HDF5 file for debugging."""
    paths = []
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            paths.append(name)
    with h5py.File(file_path, 'r') as file:
        file.visititems(visitor_func)
    return paths

def load_image_data(file_path):
    """Load 3D image data from a file path."""
    print(f"Attempting to load data from {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            paths = list_all_hdf5_paths(file_path)
            print(f"Available dataset paths in {file_path}: {paths}")
            image_dataset_paths = [path for path in paths if 'image' in path.lower()]
            if not image_dataset_paths:
                raise KeyError("No datasets containing 'image' found in the file.")
            image_stack = np.array([file[path][:] for path in image_dataset_paths])
            print(f"Loaded data shape: {image_stack.shape}")
            return image_stack
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def compare_and_calculate_effect_size(data1, data2):
    """Compare two datasets and calculate p-value and effect size."""
    normal1 = is_normal(data1)
    normal2 = is_normal(data2)
    if normal1 and normal2:
        _, p_value = ttest_ind(data1, data2)
        effect_size = calculate_cohens_d(data1, data2)
    else:
        _, p_value = mannwhitneyu(data1, data2)
        effect_size = calculate_rank_biserial(data1, data2)
    return p_value, effect_size

def save_raw_data(raw_data, output_dir, metric, data_type):
    """Save raw data to files in a subfolder for recreation of plots."""
    raw_dir = os.path.join(output_dir, 'rawcurves')
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, f"{metric}_{data_type}.txt")
    np.savetxt(file_path, raw_data, fmt='%.6f')

def run_bootstrap_analysis(image_stack1, image_stack2, n_rounds, sample_pct, output_dir, threads, options):
    """Run n bootstrap rounds, comparing subsets of the input stacks with parallel processing."""
    all_metrics = ['kurtosis', 'entropy', 'contrast', 'homogeneity', 'energy', 'correlation', 'mean', 'std_dev']
    metric_keys = [metric for metric in all_metrics if not options[f"no{metric}"]]

    results = {
        'p_values': {key: [] for key in metric_keys},
        'effect_sizes': {key: [] for key in metric_keys},
        'significance_counts': np.zeros(len(metric_keys))
    }

    for i in range(n_rounds):
        print(f"\nBootstrap round {i + 1}/{n_rounds}...")

        # Randomly sample subsets from the input stacks
        subset1 = resample(image_stack1, n_samples=int(len(image_stack1) * sample_pct))
        subset2 = resample(image_stack2, n_samples=int(len(image_stack2) * sample_pct))

        metrics1 = {key: [] for key in metric_keys}
        metrics2 = {key: [] for key in metric_keys}

        '''
        # Process subsets in parallel
        with ThreadPoolExecutor(max_workers=threads) as executor:
            subset1_results = list(executor.map(calculate_metrics_for_images, subset1))
            subset2_results = list(executor.map(calculate_metrics_for_images, subset2))
            for vol_metrics, _ in subset1_results:
                for key in metric_keys:
                    metrics1[key].extend(vol_metrics[key])
            for vol_metrics, _ in subset2_results:
                for key in metric_keys:
                    metrics2[key].extend(vol_metrics[key])
        '''
        # Process subsets in parallel
        with ThreadPoolExecutor(max_workers=threads) as executor:
            subset1_results = list(executor.map(
                calculate_metrics_for_images,
                subset1,
                [i] * len(subset1),                # Pass the iteration number
                range(len(subset1))                # Pass volume indices for subset1
            ))
            subset2_results = list(executor.map(
                calculate_metrics_for_images,
                subset2,
                [i] * len(subset2),                # Pass the iteration number
                range(len(subset2))                # Pass volume indices for subset2
            ))

        # Update metric collection
        for vol_idx, (vol_metrics, _) in enumerate(subset1_results):
            for key in metric_keys:
                metrics1[key].extend(vol_metrics[key])

        for vol_idx, (vol_metrics, _) in enumerate(subset2_results):
            for key in metric_keys:
                metrics2[key].extend(vol_metrics[key])


        # Statistical comparisons and effect size calculations
        for idx, metric in enumerate(metric_keys):
            data1 = np.array(metrics1[metric])
            data2 = np.array(metrics2[metric])
            p_value, effect_size = compare_and_calculate_effect_size(data1, data2)

            results['p_values'][metric].append(p_value)
            results['effect_sizes'][metric].append(effect_size)

            save_raw_data(results['p_values'][metric], output_dir, metric, 'pvalues')
            save_raw_data(results['effect_sizes'][metric], output_dir, metric, 'effectsize')

            if p_value < 0.05:
                results['significance_counts'][idx] += 1

    # Save final cumulative results
    with open(os.path.join(output_dir, "bootstrap_summary_results.txt"), 'w') as f:
        f.write(f"Bootstrap Summary Results (n={n_rounds}):\n\n")
        for idx, metric in enumerate(metric_keys):
            f.write(f"{metric}:\n")
            f.write(f"Significance count (p < 0.05): {int(results['significance_counts'][idx])}\n")
            f.write(f"Average p-value: {np.mean(results['p_values'][metric]):.6f}\n")
            f.write(f"Average effect size: {np.mean(results['effect_sizes'][metric]):.6f}\n\n")

    return results

'''
def visualize_results(results, output_dir, metric_keys):
    """Create visualizations for bootstrap analysis results."""
    plt.figure(figsize=(10, 6))
    num_rounds = len(next(iter(results['p_values'].values())))
    significance_fractions = results['significance_counts'] / num_rounds
    plt.bar(metric_keys, significance_fractions, color='blue', alpha=0.7)

    plt.title('Frequency of Statistical Significance')
    plt.ylabel('Fraction of Significant Iterations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/significance_frequency.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        effect_sizes = results['effect_sizes'][key]
        plt.plot(range(len(effect_sizes)), effect_sizes, label=key, marker='o')

    plt.title('Effect Size Trends Across Bootstraps')
    plt.xlabel('Iteration')
    plt.ylabel('Effect Size (Cohen\'s d or Rank-biserial)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/effect_size_trends.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        p_values = results['p_values'][key]
        plt.plot(range(len(p_values)), p_values, label=key, marker='o')

    plt.title('P-Value Trends Across Bootstraps')
    plt.xlabel('Iteration')
    plt.ylabel('P-Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/p_value_trends.png")
    plt.close()
    '''

def visualize_results(results, output_dir, metric_keys, effect_size_types):
    """Create visualizations for bootstrap analysis results."""
    # P-Value Trends
    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        p_values = results['p_values'][key]
        plt.plot(range(len(p_values)), p_values, label=key, marker='o')

    plt.axhline(y=0.05, color='gray', linestyle='--', label='p=0.05 (Significance Threshold)')
    plt.title('P-Value Trends Across Bootstraps')
    plt.xlabel('Iteration')
    plt.ylabel('P-Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/p_value_trends.png")
    plt.close()

    # Effect Size Trends
    small_threshold = 0.2
    medium_threshold = 0.5
    large_threshold = 0.8

    # Effect size distribution plot
    plt.figure(figsize=(10, 6))
    should_plot_medium = False
    should_plot_large = False

    # Threshold definitions
    thresholds = {'small': 0.1, 'medium': 0.3, 'large': 0.5}
    line_styles = {'small': (3, 3), 'medium': (5, 5), 'large': (10, 2)}

    # Plot effect sizes and track thresholds
    for key in metric_keys:
        effect_sizes = results['effect_sizes'][key]
        plt.plot(range(len(effect_sizes)), effect_sizes, label=f"{key} ({effect_size_types[key]})", marker='o')

        # Check if any effect sizes exceed medium/large thresholds
        if any(abs(val) > thresholds['medium'] for val in effect_sizes):
            should_plot_medium = True
        if any(abs(val) > thresholds['large'] for val in effect_sizes):
            should_plot_large = True

    # Add horizontal lines for thresholds
    plt.axhline(y=thresholds['small'], color='gray', linestyle=(0, line_styles['small']), linewidth=1, label='Small Effect Size (+)')
    plt.axhline(y=-thresholds['small'], color='gray', linestyle=(0, line_styles['small']), linewidth=1, label='Small Effect Size (-)')

    if should_plot_medium:
        plt.axhline(y=thresholds['medium'], color='gray', linestyle=(0, line_styles['medium']), linewidth=1, label='Medium Effect Size (+)')
        plt.axhline(y=-thresholds['medium'], color='gray', linestyle=(0, line_styles['medium']), linewidth=1, label='Medium Effect Size (-)')

    if should_plot_large:
        plt.axhline(y=thresholds['large'], color='gray', linestyle=(0, line_styles['large']), linewidth=1, label='Large Effect Size (+)')
        plt.axhline(y=-thresholds['large'], color='gray', linestyle=(0, line_styles['large']), linewidth=1, label='Large Effect Size (-)')

    # Finalize plot
    plt.title('Effect Size Trends Across Bootstraps')
    plt.xlabel('Iteration')
    plt.ylabel('Effect Size (Cohen\'s d or Rank-Biserial)')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/effect_size_trends.png")
    plt.close()




def main():
    parser = argparse.ArgumentParser(description="Bootstrap statistical analysis on one or two input stacks.")
    parser.add_argument('--input', required=True, help="Comma-separated file paths for one or two image stacks.")
    parser.add_argument('--output_dir', default='molstats_boots', help="Directory to save the outputs.")
    parser.add_argument('--n_rounds', type=int, default=100, help="Number of bootstrap iterations.")
    parser.add_argument('--sample_pct', type=float, default=0.5, help="Fraction of images to sample in each bootstrap iteration.")
    parser.add_argument('--subset', type=int, default=None, help="Number of images to use as a subset for analysis.")
    parser.add_argument('--threads', type=int, default=1, help="Number of threads to use for parallel processing.")

    # Boolean flags for disabling specific metrics
    parser.add_argument('--nokurtosis', action='store_true', help="Disable kurtosis metric.")
    parser.add_argument('--noentropy', action='store_true', help="Disable entropy metric.")
    parser.add_argument('--nocontrast', action='store_true', help="Disable contrast metric.")
    parser.add_argument('--nohomogeneity', action='store_true', help="Disable homogeneity metric.")
    parser.add_argument('--noenergy', action='store_true', help="Disable energy metric.")
    parser.add_argument('--nocorrelation', action='store_true', help="Disable correlation metric.")
    parser.add_argument('--nomean', action='store_true', help="Disable mean metric.")
    parser.add_argument('--nostd_dev', action='store_true', help="Disable std_dev metric.")

    args = parser.parse_args()

    start_time = time.time()
    log_file = os.path.join(os.getcwd(), 'molecular_stats_bootstrap.log')
    logger = setup_logger(log_file)

    logger.info(f"Command: {' '.join(os.sys.argv)}")

    output_dir = create_output_directory(args.output_dir)

    options = {
        'nokurtosis': args.nokurtosis,
        'noentropy': args.noentropy,
        'nocontrast': args.nocontrast,
        'nohomogeneity': args.nohomogeneity,
        'noenergy': args.noenergy,
        'nocorrelation': args.nocorrelation,
        'nomean': args.nomean,
        'nostd_dev': args.nostd_dev
    }

    input_files = args.input.split(',')
    if len(input_files) == 1:
        print("Single stack input detected. Splitting into two subsets for bootstrapping.")
        image_stack = load_image_data(input_files[0])

        # Apply subset if specified
        if args.subset:
            if args.subset >= len(image_stack):
                raise ValueError(f"--subset value ({args.subset}) cannot exceed the number of images ({len(image_stack)}).")
            image_stack = resample(image_stack, n_samples=args.subset, replace=False)

        mid_idx = len(image_stack) // 2
        image_stack1, image_stack2 = image_stack[:mid_idx], image_stack[mid_idx:]
    elif len(input_files) == 2:
        print("Two stack inputs detected. Performing bootstrap comparisons.")
        image_stack1 = load_image_data(input_files[0])
        image_stack2 = load_image_data(input_files[1])

        # Apply subset to each stack if specified
        if args.subset:
            min_stack_size = min(len(image_stack1), len(image_stack2))
            if args.subset >= min_stack_size:
                raise ValueError(f"--subset value ({args.subset}) cannot exceed the size of the smallest stack ({min_stack_size}).")
            image_stack1 = resample(image_stack1, n_samples=args.subset, replace=False)
            image_stack2 = resample(image_stack2, n_samples=args.subset, replace=False)
    else:
        raise ValueError("Invalid number of input files. Provide one or two stacks.")

    '''
    results = run_bootstrap_analysis(image_stack1, image_stack2, args.n_rounds, args.sample_pct, output_dir, args.threads, options)
    metric_keys = [key for key in ['contrast', 'correlation', 'energy', 'homogeneity', 'kurtosis', 'entropy', 'mean', 'std_dev'] if not options[f"no{key}"]]
    visualize_results(results, output_dir, metric_keys)

    elapsed_time = time.time() - start_time
    logger.info(f"Execution completed in {elapsed_time:.2f} seconds.")

    print(f"Bootstrap analysis completed. Results saved in {output_dir}.")
    '''
    metric_keys = [key for key in ['contrast', 'correlation', 'energy', 'homogeneity', 'kurtosis', 'entropy', 'mean', 'std_dev'] if not options[f"no{key}"]]

    # Determine which metrics use Cohen's d or Rank-Biserial based on normality
    effect_size_types = {key: "Cohen's d" if key in ['mean', 'std_dev'] else "Rank-Biserial" for key in metric_keys}

    results = run_bootstrap_analysis(image_stack1, image_stack2, args.n_rounds, args.sample_pct, output_dir, args.threads, options)

    # Visualize results
    visualize_results(results, output_dir, metric_keys, effect_size_types)

    elapsed_time = time.time() - start_time
    logger.info(f"Execution completed in {elapsed_time:.2f} seconds.")
    print(f"Bootstrap analysis completed. Results saved in {output_dir}.")



if __name__ == '__main__':
    main()





'''
#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2024; last modification: Dec/13/2024

import h5py
import numpy as np
import argparse
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import matplotlib.pyplot as plt
from sklearn.utils import resample
from molecular_stats import calculate_metrics_for_images, is_normal, calculate_cohens_d, calculate_rank_biserial, create_output_directory
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor

def setup_logger(log_file):
    """Set up a logger to record script execution details."""
    logger = logging.getLogger('molecular_stats_bootstrap')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def list_all_hdf5_paths(file_path):
    """List all paths in an HDF5 file for debugging."""
    paths = []
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            paths.append(name)
    with h5py.File(file_path, 'r') as file:
        file.visititems(visitor_func)
    return paths

def load_image_data(file_path):
    """Load 3D image data from a file path."""
    print(f"Attempting to load data from {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            paths = list_all_hdf5_paths(file_path)
            print(f"Available dataset paths in {file_path}: {paths}")
            image_dataset_paths = [path for path in paths if 'image' in path.lower()]
            if not image_dataset_paths:
                raise KeyError("No datasets containing 'image' found in the file.")
            image_stack = np.array([file[path][:] for path in image_dataset_paths])
            print(f"Loaded data shape: {image_stack.shape}")
            return image_stack
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def compare_and_calculate_effect_size(data1, data2):
    """Compare two datasets and calculate p-value and effect size."""
    normal1 = is_normal(data1)
    normal2 = is_normal(data2)
    if normal1 and normal2:
        _, p_value = ttest_ind(data1, data2)
        effect_size = calculate_cohens_d(data1, data2)
    else:
        _, p_value = mannwhitneyu(data1, data2)
        effect_size = calculate_rank_biserial(data1, data2)
    return p_value, effect_size

def run_bootstrap_analysis(image_stack1, image_stack2, n_rounds, sample_pct, convergence_delta, output_dir, threads):
    """Run n bootstrap rounds, comparing subsets of the input stacks with parallel processing."""
    metric_keys = ['kurtosis', 'entropy', 'contrast', 'homogeneity', 'energy', 'correlation', 'mean', 'std_dev']
    results = {
        'p_values': {key: [] for key in metric_keys},
        'effect_sizes': {key: [] for key in metric_keys},
        'significance_counts': np.zeros(len(metric_keys))
    }

    previous_effect_sizes = {key: [] for key in metric_keys}

    for i in range(n_rounds):
        print(f"\nBootstrap round {i + 1}/{n_rounds}...")

        # Randomly sample subsets from the input stacks
        subset1 = resample(image_stack1, n_samples=int(len(image_stack1) * sample_pct))
        subset2 = resample(image_stack2, n_samples=int(len(image_stack2) * sample_pct))

        metrics1 = {key: [] for key in metric_keys}
        metrics2 = {key: [] for key in metric_keys}

        # Process subsets in parallel
        with ThreadPoolExecutor(max_workers=threads) as executor:
            subset1_results = list(executor.map(calculate_metrics_for_images, subset1))
            subset2_results = list(executor.map(calculate_metrics_for_images, subset2))
            for vol_metrics, _ in subset1_results:
                for key in metric_keys:
                    metrics1[key].extend(vol_metrics[key])
            for vol_metrics, _ in subset2_results:
                for key in metric_keys:
                    metrics2[key].extend(vol_metrics[key])

        # Statistical comparisons and effect size calculations
        #converged = True
        converged=False
        for idx, metric in enumerate(metric_keys):
            data1 = np.array(metrics1[metric])
            data2 = np.array(metrics2[metric])
            p_value, effect_size = compare_and_calculate_effect_size(data1, data2)

            results['p_values'][metric].append(p_value)
            results['effect_sizes'][metric].append(effect_size)

            if p_value < 0.05:
                results['significance_counts'][idx] += 1

            # Check convergence
            #if previous_effect_sizes[metric]:
            #    delta = abs(effect_size - np.mean(previous_effect_sizes[metric]))
            #    if delta < convergence_delta:
            #        converged = False
            previous_effect_sizes[metric].append(effect_size)

        if converged:
            print(f"Convergence achieved after {i + 1} iterations.")
            break

    # Save final cumulative results
    with open(os.path.join(output_dir, "bootstrap_summary_results.txt"), 'w') as f:
        f.write(f"Bootstrap Summary Results (n={n_rounds}):\n\n")
        for idx, metric in enumerate(metric_keys):
            f.write(f"{metric}:\n")
            f.write(f"Significance count (p < 0.05): {int(results['significance_counts'][idx])}\n")
            f.write(f"Average p-value: {np.mean(results['p_values'][metric]):.6f}\n")
            f.write(f"Average effect size: {np.mean(results['effect_sizes'][metric]):.6f}\n\n")

    return results

def visualize_results(results, output_dir, metric_keys):
    """Create visualizations for bootstrap analysis results."""
    plt.figure(figsize=(10, 6))
    num_rounds = len(next(iter(results['p_values'].values())))
    significance_fractions = results['significance_counts'] / num_rounds
    plt.bar(metric_keys, significance_fractions, color='blue', alpha=0.7)

    plt.title('Frequency of Statistical Significance')
    plt.ylabel('Fraction of Significant Iterations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/significance_frequency.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        effect_sizes = results['effect_sizes'][key]
        plt.plot(effect_sizes, label=key, marker='o')

    plt.title('Effect Size Trends Across Bootstraps')
    plt.xlabel('Iteration')
    plt.ylabel('Effect Size (Cohen\'s d or Rank-biserial)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/effect_size_trends.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        p_values = results['p_values'][key]
        plt.plot(p_values, label=key, marker='o')

    plt.title('P-Value Trends Across Bootstraps')
    plt.xlabel('Iteration')
    plt.ylabel('P-Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/p_value_trends.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Bootstrap statistical analysis on one or two input stacks.")
    parser.add_argument('--input', required=True, help="Comma-separated file paths for one or two image stacks.")
    parser.add_argument('--output_dir', default='molstats_boots', help="Directory to save the outputs.")
    parser.add_argument('--n_rounds', type=int, default=100, help="Number of bootstrap iterations.")
    parser.add_argument('--sample_pct', type=float, default=0.5, help="Fraction of images to sample in each bootstrap iteration (default: 0.5).")
    parser.add_argument('--convergence_delta', type=float, default=0.01, help="Convergence threshold for effect size changes (default: 0.01).")
    parser.add_argument('--threads', type=int, default=1, help="Number of threads to use for parallel processing.")
    args = parser.parse_args()

    start_time = time.time()
    log_file = os.path.join(os.getcwd(), 'molecular_stats_bootstrap.log')
    logger = setup_logger(log_file)

    logger.info(f"Command: {' '.join(os.sys.argv)}")

    output_dir = create_output_directory(args.output_dir)

    input_files = args.input.split(',')
    if len(input_files) == 1:
        print("Single stack input detected. Splitting into two subsets for bootstrapping.")
        image_stack = load_image_data(input_files[0])
        mid_idx = len(image_stack) // 2
        image_stack1, image_stack2 = image_stack[:mid_idx], image_stack[mid_idx:]
    elif len(input_files) == 2:
        print("Two stack inputs detected. Performing bootstrap comparisons.")
        image_stack1 = load_image_data(input_files[0])
        image_stack2 = load_image_data(input_files[1])
    else:
        raise ValueError("Invalid number of input files. Provide one or two stacks.")

    results = run_bootstrap_analysis(image_stack1, image_stack2, args.n_rounds, args.sample_pct, args.convergence_delta, output_dir, args.threads)
    metric_keys = ['contrast', 'correlation', 'energy', 'homogeneity', 'kurtosis', 'entropy', 'mean', 'std_dev']
    visualize_results(results, output_dir, metric_keys)

    elapsed_time = time.time() - start_time
    logger.info(f"Execution completed in {elapsed_time:.2f} seconds.")

    print(f"Bootstrap analysis completed. Results saved in {output_dir}.")

if __name__ == '__main__':
    main()

'''