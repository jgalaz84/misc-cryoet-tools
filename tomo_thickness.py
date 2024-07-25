#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import mrcfile
import h5py
from scipy.ndimage import gaussian_filter, sobel
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
import multiprocessing
import psutil

def main():
    print("Starting the script...")
    parser = argparse.ArgumentParser(description="Estimates ice thickness from tomograms")
    parser.add_argument("--align_flat", action="store_true", default=False, help="Align the tomogram to make the region containing the specimen flat")
    parser.add_argument("--bin", type=int, default=1, help="Downsample the tomograms by the given integer factor (default: 1, no downsampling)")
    parser.add_argument("--entropy", action="store_true", default=False, help="Include entropy in the analysis")
    parser.add_argument("--gradient", action="store_true", default=False, help="Include gradient magnitude in the analysis")
    parser.add_argument("--input", type=str, help="Comma-separated .mrc/.hdf/.rec files to process")
    parser.add_argument("--inputdir", type=str, default=None, help="Input directory with multiple .mrc/.hdf/.rec files to process (default: None)")
    parser.add_argument("--mean", action="store_true", default=False, help="Include mean density in the analysis")
    parser.add_argument("--meanfilter", type=float, default=0, help="Apply low-pass filter to mean density, expressed as a fraction of Nyquist (0.5 is Nyquist, 1 is half Nyquist, etc.)")
    parser.add_argument("--min_thickness", type=int, default=10, help="Minimum thickness between the first and second minimum (default: 10)")
    parser.add_argument("--padz", type=int, default=0, help="Pad the trim boundaries by the given number of pixels (default: 0, no padding)")
    parser.add_argument("--path", type=str, default='tomo_thickness_00', help="Directory to store results")
    parser.add_argument("--savetrim", action="store_true", default=False, help="Save the trimmed versions of the tomograms")
    parser.add_argument("--skewness", action="store_true", default=False, help="Include skewness in the analysis")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for processing (default: 1)")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbose level [0-9]")

    args = parser.parse_args()
    print("Arguments parsed:", args)

    max_threads = multiprocessing.cpu_count()
    if args.threads > max_threads:
        args.threads = max_threads

    available_memory = psutil.virtual_memory().available / (1024**3)  # in GB
    estimated_memory_per_thread = 2  # Estimate 2 GB per thread for safety
    max_threads_based_on_memory = int(available_memory * 0.75 / estimated_memory_per_thread)
    if args.threads > max_threads_based_on_memory:
        args.threads = max_threads_based_on_memory

    if not args.input and not args.inputdir:
        print("\nERROR: Either one of --input or --inputdir required.")
        sys.exit(1)
    
    inputs = []
    if args.input:
        inputs = args.input.split(',')
    elif args.inputdir:
        inputs = [os.path.join(args.inputdir, f) for f in os.listdir(args.inputdir) if f.endswith(('.mrc', '.hdf', '.rec'))]

    for file in inputs:
        if not os.path.isfile(file):
            print(f"ERROR: File not found - {file}")
            sys.exit(1)

    base_path = args.path.rstrip('_00')
    counter = 0
    output_path = f"{base_path}_{counter:02d}"
    while os.path.exists(output_path):
        counter += 1
        output_path = f"{base_path}_{counter:02d}"
    args.path = output_path
    os.makedirs(args.path)

    print("Inputs detected:", inputs)
    
    metrics_paths = {}
    if args.mean:
        metrics_paths['means'] = os.path.join(args.path, 'means')
    metrics_paths['stds'] = os.path.join(args.path, 'stds')
    if args.skewness:
        metrics_paths['skews'] = os.path.join(args.path, 'skews')
    metrics_paths['kurtosis'] = os.path.join(args.path, 'kurtosis')
    if args.gradient:
        metrics_paths['gradients'] = os.path.join(args.path, 'gradients')
    if args.entropy:
        metrics_paths['entropy'] = os.path.join(args.path, 'entropy')

    for path in metrics_paths.values():
        os.makedirs(path, exist_ok=True)

    print("Starting multiprocessing with", args.threads, "threads")
    pool = multiprocessing.Pool(processes=args.threads)
    pool.map(process_tomogram, [(t, args, metrics_paths) for t in inputs])
    pool.close()
    pool.join()

    
def process_tomogram(params):
    t, args, metrics_paths = params
    print("Processing tomogram:", t)
    
    if not os.path.isfile(t):
        print(f"ERROR: File not found during processing - {t}")
        return

    if t.endswith('.hdf') or t.endswith('.h5'):
        with h5py.File(t, 'r') as f:
            dataset_name = find_dataset(f)
            tomogram = f[dataset_name][()]
    else:
        with mrcfile.open(t, permissive=True) as mrc:
            tomogram = mrc.data

    if tomogram is None or tomogram.ndim != 3:
        print(f"ERROR: Tomogram data is invalid or not 3D - {t}")
        return

    if args.bin > 1:
        downsampled_tomogram = fourier_crop(tomogram, args.bin)
    else:
        downsampled_tomogram = tomogram
    
    means, stds, skews, kurt, gradients, entropies = compute_statistics(downsampled_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
    save_statistics(means, stds, skews, kurt, gradients, entropies, t, metrics_paths)
    plot_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    
    trim_bounds_std, thickness_std = compute_trim_bounds_from_std(stds, args.min_thickness)
    trim_bounds_kurt, thickness_kurt = compute_trim_bounds_from_kurtosis(kurt, args.min_thickness)
    trim_bounds_combined, thickness_combined = compute_trim_bounds_combined(stds, kurt, args.min_thickness)

    avg_trim_bounds = ((trim_bounds_std[0] + trim_bounds_kurt[0]) // 2, (trim_bounds_std[1] + trim_bounds_kurt[1]) // 2)
    avg_thickness = (thickness_std + thickness_kurt) // 2

    trim_bounds_std = (trim_bounds_std[0] * args.bin, trim_bounds_std[1] * args.bin)
    trim_bounds_kurt = (trim_bounds_kurt[0] * args.bin, trim_bounds_kurt[1] * args.bin)
    trim_bounds_combined = (trim_bounds_combined[0] * args.bin, trim_bounds_combined[1] * args.bin)
    avg_trim_bounds = (avg_trim_bounds[0] * args.bin, avg_trim_bounds[1] * args.bin)

    save_trim_bounds(avg_trim_bounds, t, args.path, "trimz_avg")
    save_trim_bounds(trim_bounds_combined, t, args.path, "trimz_combined")
    save_thicknesses(thickness_std * args.bin, thickness_kurt * args.bin, thickness_combined * args.bin, avg_thickness * args.bin, t, args.path)
    
    if args.savetrim:
        padded_trim_bounds = (max(0, avg_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_trim_bounds[1] + args.padz))
        padded_trim_bounds_combined = (max(0, trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, trim_bounds_combined[1] + args.padz))

        trimmed_tomogram_avg = trim_tomogram(tomogram, padded_trim_bounds)
        trimmed_tomogram_combined = trim_tomogram(tomogram, padded_trim_bounds_combined)

        save_trimmed_tomogram(trimmed_tomogram_avg, t, args.path, suffix="_trimz_avg")
        save_trimmed_tomogram(trimmed_tomogram_combined, t, args.path, suffix="_trimz_combined")

    if args.align_flat:
        rotated_tomogram, rotation_matrix = align_tomogram_with_pca(downsampled_tomogram)
        save_rotated_tomogram(rotated_tomogram, t, args.path)
        save_rotation_matrix(rotation_matrix, t, args.path)
        
        rotated_means, rotated_stds, rotated_skews, rotated_kurt, rotated_gradients, rotated_entropies = compute_statistics(rotated_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
        rotated_trim_bounds_std, rotated_thickness_std = compute_trim_bounds_from_std(rotated_stds, args.min_thickness)
        rotated_trim_bounds_kurt, rotated_thickness_kurt = compute_trim_bounds_from_kurtosis(rotated_kurt, args.min_thickness)
        rotated_trim_bounds_combined, rotated_thickness_combined = compute_trim_bounds_combined(rotated_stds, rotated_kurt, args.min_thickness)

        avg_rotated_trim_bounds = ((rotated_trim_bounds_std[0] + rotated_trim_bounds_kurt[0]) // 2, (rotated_trim_bounds_std[1] + rotated_trim_bounds_kurt[1]) // 2)
        avg_rotated_thickness = (rotated_thickness_std + rotated_thickness_kurt) // 2

        rotated_trim_bounds_std = (rotated_trim_bounds_std[0] * args.bin, rotated_trim_bounds_std[1] * args.bin)
        rotated_trim_bounds_kurt = (rotated_trim_bounds_kurt[0] * args.bin, rotated_trim_bounds_kurt[1] * args.bin)
        rotated_trim_bounds_combined = (rotated_trim_bounds_combined[0] * args.bin, rotated_trim_bounds_combined[1] * args.bin)
        avg_rotated_trim_bounds = (avg_rotated_trim_bounds[0] * args.bin, avg_rotated_trim_bounds[1] * args.bin)

        save_trim_bounds(avg_rotated_trim_bounds, t, args.path, "trimz_aligned_avg")
        save_trim_bounds(rotated_trim_bounds_combined, t, args.path, "trimz_aligned_combined")
        save_thicknesses(rotated_thickness_std * args.bin, rotated_thickness_kurt * args.bin, rotated_thickness_combined * args.bin, avg_rotated_thickness * args.bin, t, args.path, suffix="_aligned")
        
        if args.savetrim:
            padded_rotated_trim_bounds = (max(0, avg_rotated_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_rotated_trim_bounds[1] + args.padz))
            padded_rotated_trim_bounds_combined = (max(0, rotated_trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, rotated_trim_bounds_combined[1] + args.padz))

            trimmed_rotated_tomogram_avg = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds)
            trimmed_rotated_tomogram_combined = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds_combined)

            save_trimmed_tomogram(trimmed_rotated_tomogram_avg, t, args.path, suffix="_ali_trimz_avg")
            save_trimmed_tomogram(trimmed_rotated_tomogram_combined, t, args.path, suffix="_ali_trimz_combined")


def find_dataset(h5file):
    dataset_name = None
    def find_name(name, obj):
        nonlocal dataset_name
        if isinstance(obj, h5py.Dataset) and dataset_name is None:
            dataset_name = name
    h5file.visititems(find_name)
    return dataset_name


def fourier_crop(tomogram, bin_factor):
    fft_data = fftshift(fftn(tomogram))
    crop_size = [dim // bin_factor for dim in tomogram.shape]
    cropped_fft_data = fft_data[
        (tomogram.shape[0] // 2 - crop_size[0] // 2):(tomogram.shape[0] // 2 + crop_size[0] // 2),
        (tomogram.shape[1] // 2 - crop_size[1] // 2):(tomogram.shape[1] // 2 + crop_size[1] // 2),
        (tomogram.shape[2] // 2 - crop_size[2] // 2):(tomogram.shape[2] // 2 + crop_size[2] // 2)
    ]
    cropped_fft_data = ifftshift(cropped_fft_data)
    cropped_tomogram = np.real(ifftn(cropped_fft_data))
    return cropped_tomogram

def compute_statistics(tomogram, meanfilter, include_gradient, include_mean, include_skewness, include_entropy):
    means = None
    if include_mean:
        means = np.mean(tomogram, axis=(1, 2))
        if meanfilter > 0:
            nyquist = 0.5
            cutoff = nyquist * meanfilter
            means = gaussian_filter(means, sigma=cutoff)
    
    stds = np.std(tomogram, axis=(1, 2))
    skews = None
    if include_skewness:
        skews = np.zeros(tomogram.shape[0])
        for i in range(tomogram.shape[0]):
            slice_data = tomogram[i].flatten()
            if np.all(slice_data == 0):
                skews[i] = 0
            else:
                skews[i] = np.nan_to_num(skew(slice_data))

    kurt = np.zeros(tomogram.shape[0])
    for i in range(tomogram.shape[0]):
        slice_data = tomogram[i].flatten()
        if np.all(slice_data == 0):
            kurt[i] = 0
        else:
            kurt[i] = np.nan_to_num(kurtosis(slice_data))

    gradients = []
    if include_gradient:
        gradients = [np.mean(sobel(gaussian_filter(tomogram[i], sigma=2))) for i in range(tomogram.shape[0])]
    
    entropies = None
    if include_entropy:
        entropies = []
        for i in range(tomogram.shape[0]):
            try:
                entropy_value = np.mean(entropy(img_as_ubyte(rescale_intensity(tomogram[i], in_range='float', out_range=(0, 1))), disk(5)))
                entropies.append(entropy_value)
            except:
                entropies.append(0)
        entropies = np.array(entropies)
    
    return means, stds, skews, kurt, gradients, entropies

def save_statistics(means, stds, skews, kurt, gradients, entropies, filename, metrics_paths):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    if means is not None:
        np.savetxt(os.path.join(metrics_paths['means'], f'{base}_means.txt'), means)
    np.savetxt(os.path.join(metrics_paths['stds'], f'{base}_stds.txt'), stds)
    if skews is not None:
        np.savetxt(os.path.join(metrics_paths['skews'], f'{base}_skews.txt'), skews)
    np.savetxt(os.path.join(metrics_paths['kurtosis'], f'{base}_kurtosis.txt'), kurt)
    if gradients:
        np.savetxt(os.path.join(metrics_paths['gradients'], f'{base}_gradients.txt'), gradients)
    if entropies is not None:
        np.savetxt(os.path.join(metrics_paths['entropy'], f'{base}_entropy.txt'), entropies)

def plot_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    # Normalize metrics
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')
    
    normalized_metrics = []
    for metric in metrics:
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        normalized_metrics.append(normalized_metric)

    plt.figure()
    for metric, label in zip(normalized_metrics, labels):
        plt.plot(metric, label=label)
    
    # Calculate and plot combined metric
    combined_metric = np.sum(normalized_metrics, axis=0)
    combined_metric = np.nan_to_num(combined_metric)  # Replace nan with zero
    if np.max(combined_metric) != 0:
        combined_metric /= np.max(combined_metric)  # Normalize combined metric
    plt.plot(combined_metric, label='Combined Metric', linestyle='--', color='black')

    plt.xlabel('Z Slice')
    plt.ylabel('Normalized Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.title(f'Statistics for {base}')
    plt.savefig(os.path.join(output_path, f'{base}_statistics.png'), bbox_inches='tight')
    plt.close()

def plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for ax, metric, label in zip(axs, metrics, labels):
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        ax.plot(normalized_metric)
        ax.set_title(label)
        ax.set_xlabel('Z Slice')
        ax.set_ylabel('Normalized Value')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base}_separate_statistics.png'))
    plt.close()

def compute_trim_bounds_from_std(stds, min_thickness):
    gradient_of_stds = np.gradient(stds)
    maxima = np.where((np.r_[True, gradient_of_stds[1:] > gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] > gradient_of_stds[1:], True]))[0]
    minima = np.where((np.r_[True, gradient_of_stds[1:] < gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] < gradient_of_stds[1:], True]))[0]

    # Find the first large maximum and the last large minimum
    start = maxima[0] if len(maxima) > 0 else 0
    end = minima[-1] if len(minima) > 0 else len(stds) - 1

    return (start, end), end - start

def compute_trim_bounds_from_kurtosis(kurt, min_thickness):
    kurt_list = list(kurt)
    max_index = kurt_list.index(max(kurt_list))

    kurt_list_left_of_max = kurt_list[:max_index]
    kurt_list_right_of_max = kurt_list[max_index:]

    # Find the first minimum to the left of the maximum
    left_min_index = len(kurt_list_left_of_max) - 1 - kurt_list_left_of_max[::-1].index(min(kurt_list_left_of_max))
    
    # Find the first minimum to the right of the maximum
    right_min_index = kurt_list_right_of_max.index(min(kurt_list_right_of_max)) + max_index

    # Ensure the minima are not too close to each other
    if right_min_index - left_min_index < min_thickness:
        if len(kurt_list_left_of_max) > 1:
            kurt_list_left_of_max[left_min_index] = max(kurt_list)
            left_min_index = len(kurt_list_left_of_max) - 1 - kurt_list_left_of_max[::-1].index(min(kurt_list_left_of_max))
        if len(kurt_list_right_of_max) > 1:
            kurt_list_right_of_max[right_min_index - max_index] = max(kurt_list)
            right_min_index = kurt_list_right_of_max.index(min(kurt_list_right_of_max)) + max_index

    return (left_min_index, right_min_index), right_min_index - left_min_index

def compute_trim_bounds_combined(stds, kurt, min_thickness):
    combined_metric = stds + kurt
    combined_list = list(combined_metric)
    max_index = combined_list.index(max(combined_list))

    combined_list_left_of_max = combined_list[:max_index]
    combined_list_right_of_max = combined_list[max_index:]

    # Find the first minimum to the left of the maximum
    left_min_index = len(combined_list_left_of_max) - 1 - combined_list_left_of_max[::-1].index(min(combined_list_left_of_max))
    
    # Find the first minimum to the right of the maximum
    right_min_index = combined_list_right_of_max.index(min(combined_list_right_of_max)) + max_index

    # Ensure the minima are not too close to each other
    if right_min_index - left_min_index < min_thickness:
        if len(combined_list_left_of_max) > 1:
            combined_list_left_of_max[left_min_index] = max(combined_list)
            left_min_index = len(combined_list_left_of_max) - 1 - combined_list_left_of_max[::-1].index(min(combined_list_left_of_max))
        if len(combined_list_right_of_max) > 1:
            combined_list_right_of_max[right_min_index - max_index] = max(combined_list)
            right_min_index = combined_list_right_of_max.index(min(combined_list_right_of_max)) + max_index

    return (left_min_index, right_min_index), right_min_index - left_min_index

def save_trim_bounds(bounds, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}_{suffix}.txt')
    with open(output_file, 'w') as f:
        f.write(f"{bounds[0]} {bounds[1]}\n")

def save_thicknesses(thickness_std, thickness_kurt, thickness_combined, avg_thickness, filename, output_path, suffix=""):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}_thicknesses{suffix}.txt')
    with open(output_file, 'w') as f:
        f.write(f"STD Thickness: {thickness_std}\n")
        f.write(f"Kurtosis Thickness: {thickness_kurt}\n")
        f.write(f"Combined Thickness: {thickness_combined}\n")
        f.write(f"Average Thickness: {avg_thickness}\n")

def trim_tomogram(tomogram, bounds):
    return tomogram[bounds[0]:bounds[1]]

def save_trimmed_tomogram(trimmed_tomogram, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}{suffix}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(trimmed_tomogram.astype(np.float32))

def align_tomogram_with_pca(tomogram):
    # Downsample the tomogram for alignment
    downsampled = tomogram[::8, ::8, ::8]
    
    # Flatten tomogram data for PCA
    nz, ny, nx = downsampled.shape
    coords = np.array(np.meshgrid(range(nz), range(ny), range(nx))).reshape(3, -1).T
    intensities = downsampled.flatten()
    pca = PCA(n_components=2)
    pca.fit(coords, sample_weight=intensities)
    rotation_matrix = pca.components_
    
    # Rotate the original tomogram
    rotated = apply_rotation(tomogram, rotation_matrix)
    return rotated, rotation_matrix

def apply_rotation(tomogram, rotation_matrix):
    new_shape = tomogram.shape
    new_tomogram = np.zeros(new_shape)
    for z in range(new_shape[0]):
        for y in range(new_shape[1]):
            for x in range(new_shape[2]):
                coords = np.array([z, y, x])
                new_coords = np.dot(rotation_matrix, coords - np.array(new_shape) / 2) + np.array(new_shape) / 2
                if np.all(new_coords >= 0) and np.all(new_coords < new_shape):
                    new_tomogram[z, y, x] = tomogram[tuple(new_coords.astype(int))]
    return new_tomogram

def save_rotated_tomogram(rotated_tomogram, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram.astype(np.float32))

def save_rotation_matrix(rotation_matrix, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}_rotation_matrix.txt')
    np.savetxt(output_file, rotation_matrix)

if __name__ == "__main__":
    main()




'''
WORKED july 25th 2024 but didn't handle HDF
#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import mrcfile
from scipy.ndimage import gaussian_filter, sobel
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
import multiprocessing
import psutil

def main():
    print("Starting the script...")
    parser = argparse.ArgumentParser(description="Estimates ice thickness from tomograms")
    parser.add_argument("--align_flat", action="store_true", default=False, help="Align the tomogram to make the region containing the specimen flat")
    parser.add_argument("--bin", type=int, default=1, help="Downsample the tomograms by the given integer factor (default: 1, no downsampling)")
    parser.add_argument("--entropy", action="store_true", default=False, help="Include entropy in the analysis")
    parser.add_argument("--gradient", action="store_true", default=False, help="Include gradient magnitude in the analysis")
    parser.add_argument("--input", type=str, help="Comma-separated .mrc/.hdf/.rec files to process")
    parser.add_argument("--inputdir", type=str, default='.', help="Input directory with multiple .mrc/.hdf/.rec files to process (default: current directory)")
    parser.add_argument("--mean", action="store_true", default=False, help="Include mean density in the analysis")
    parser.add_argument("--meanfilter", type=float, default=0, help="Apply low-pass filter to mean density, expressed as a fraction of Nyquist (0.5 is Nyquist, 1 is half Nyquist, etc.)")
    parser.add_argument("--min_thickness", type=int, default=None, help="Minimum thickness to ensure the two minima are not too close (default: half of tomogram size in z)")
    parser.add_argument("--padz", type=int, default=0, help="Pad the trim boundaries by the given number of pixels (default: 0, no padding)")
    parser.add_argument("--path", type=str, default='tomo_thickness_00', help="Directory to store results")
    parser.add_argument("--savetrim", action="store_true", default=False, help="Save the trimmed versions of the tomograms")
    parser.add_argument("--skewness", action="store_true", default=False, help="Include skewness in the analysis")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for processing (default: 1)")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbose level [0-9]")

    args = parser.parse_args()
    print("Arguments parsed:", args)

    max_threads = multiprocessing.cpu_count()
    if args.threads > max_threads:
        args.threads = max_threads

    available_memory = psutil.virtual_memory().available / (1024**3)  # in GB
    estimated_memory_per_thread = 2  # Estimate 2 GB per thread for safety
    max_threads_based_on_memory = int(available_memory * 0.75 / estimated_memory_per_thread)
    if args.threads > max_threads_based_on_memory:
        args.threads = max_threads_based_on_memory

    if not args.input and not args.inputdir:
        print("\nERROR: Either one of --input or --inputdir required.")
        sys.exit(1)
    
    inputs = []
    if args.inputdir:
        inputs = [os.path.join(args.inputdir, f) for f in os.listdir(args.inputdir) if f.endswith(('.mrc', '.hdf', '.rec'))]
    elif args.input:
        inputs = args.input.split(',')
    inputs.sort()

    for file in inputs:
        if not os.path.isfile(file):
            print(f"ERROR: File not found - {file}")
            sys.exit(1)

    base_path = args.path.rstrip('_00')
    counter = 0
    output_path = f"{base_path}_{counter:02d}"
    while os.path.exists(output_path):
        counter += 1
        output_path = f"{base_path}_{counter:02d}"
    args.path = output_path
    os.makedirs(args.path)

    print("Inputs detected:", inputs)
    
    metrics_paths = {}
    if args.mean:
        metrics_paths['means'] = os.path.join(args.path, 'means')
    metrics_paths['stds'] = os.path.join(args.path, 'stds')
    if args.skewness:
        metrics_paths['skews'] = os.path.join(args.path, 'skews')
    metrics_paths['kurtosis'] = os.path.join(args.path, 'kurtosis')
    if args.gradient:
        metrics_paths['gradients'] = os.path.join(args.path, 'gradients')
    if args.entropy:
        metrics_paths['entropy'] = os.path.join(args.path, 'entropy')

    for path in metrics_paths.values():
        os.makedirs(path, exist_ok=True)

    print("Starting multiprocessing with", args.threads, "threads")
    pool = multiprocessing.Pool(processes=args.threads)
    pool.map(process_tomogram, [(t, args, metrics_paths) for t in inputs])
    pool.close()
    pool.join()

def process_tomogram(params):
    t, args, metrics_paths = params
    print("Processing tomogram:", t)
    
    if not os.path.isfile(t):
        print(f"ERROR: File not found during processing - {t}")
        return
    
    with mrcfile.open(t, permissive=True) as mrc:
        tomogram = mrc.data

    if tomogram is None or tomogram.ndim != 3:
        print(f"ERROR: Tomogram data is invalid or not 3D - {t}")
        return

    if args.bin > 1:
        downsampled_tomogram = fourier_crop(tomogram, args.bin)
    else:
        downsampled_tomogram = tomogram
    
    means, stds, skews, kurt, gradients, entropies = compute_statistics(downsampled_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
    save_statistics(means, stds, skews, kurt, gradients, entropies, t, metrics_paths)
    plot_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    
    min_thickness = args.min_thickness if args.min_thickness is not None else downsampled_tomogram.shape[0] // 2

    trim_bounds_std, thickness_std = compute_trim_bounds_from_std(stds)
    trim_bounds_kurt, thickness_kurt = compute_trim_bounds_from_kurtosis(kurt, min_thickness)
    trim_bounds_combined, thickness_combined = compute_trim_bounds_combined(stds, kurt, min_thickness)
    avg_trim_bounds = ((trim_bounds_std[0] + trim_bounds_kurt[0]) // 2, (trim_bounds_std[1] + trim_bounds_kurt[1]) // 2)

    print("\n\n\nBEFORE compensating for binning, trim_bounds_std={}, trim_bounds_kurt={}, trim_bounds_combined={}, avg_trim_bounds={}".format(trim_bounds_std, trim_bounds_kurt, trim_bounds_combined, avg_trim_bounds))
    print("\n\n\nBEFORE compensating for binning, thickness_std={}, thickness_kurt={}, thickness_combined={}".format(thickness_std, thickness_kurt, thickness_combined))

    trim_bounds_std = (trim_bounds_std[0] * args.bin, trim_bounds_std[1] * args.bin)
    trim_bounds_kurt = (trim_bounds_kurt[0] * args.bin, trim_bounds_kurt[1] * args.bin)
    trim_bounds_combined = (trim_bounds_combined[0] * args.bin, trim_bounds_combined[1] * args.bin)
    avg_trim_bounds = (avg_trim_bounds[0] * args.bin, avg_trim_bounds[1] * args.bin)

    print("\n\n\nBEFORE compensating for binning, trim_bounds_std={}, trim_bounds_kurt={}, trim_bounds_combined={}".format(trim_bounds_std, trim_bounds_kurt, trim_bounds_combined))

    save_trim_bounds(avg_trim_bounds, t, args.path, "trimz_avg")
    save_trim_bounds(trim_bounds_combined, t, args.path, "trimz_combined")
    
    thickness_std *= args.bin
    thickness_kurt *= args.bin
    thickness_combined *= args.bin
    thickness_avg = (thickness_std + thickness_kurt) // 2

    print("\n\n\nAFTER compensating for binning, thickness_std={}, thickness_kurt={}, thickness_combined={}".format(thickness_std, thickness_kurt, thickness_combined))

    save_thicknesses(thickness_std, thickness_kurt, thickness_combined, thickness_avg, t, args.path)
    
    if args.savetrim:
        padded_trim_bounds = (max(0, avg_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_trim_bounds[1] + args.padz))
        padded_trim_bounds_combined = (max(0, trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, trim_bounds_combined[1] + args.padz))

        trimmed_tomogram_avg = trim_tomogram(tomogram, padded_trim_bounds)
        trimmed_tomogram_combined = trim_tomogram(tomogram, padded_trim_bounds_combined)

        save_trimmed_tomogram(trimmed_tomogram_avg, t, args.path, suffix="_trimz_avg")
        save_trimmed_tomogram(trimmed_tomogram_combined, t, args.path, suffix="_trimz_combined")

    if args.align_flat:
        rotated_tomogram, rotation_matrix = align_tomogram_with_pca(downsampled_tomogram)
        save_rotated_tomogram(rotated_tomogram, t, args.path)
        save_rotation_matrix(rotation_matrix, t, args.path)
        
        rotated_means, rotated_stds, rotated_skews, rotated_kurt, rotated_gradients, rotated_entropies = compute_statistics(rotated_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
        rotated_trim_bounds_std, rotated_thickness_std = compute_trim_bounds_from_std(rotated_stds)
        rotated_trim_bounds_kurt, rotated_thickness_kurt = compute_trim_bounds_from_kurtosis(rotated_kurt, min_thickness)
        rotated_trim_bounds_combined, rotated_thickness_combined = compute_trim_bounds_combined(rotated_stds, rotated_kurt, min_thickness)

        avg_rotated_trim_bounds = ((rotated_trim_bounds_std[0] + rotated_trim_bounds_kurt[0]) // 2, (rotated_trim_bounds_std[1] + rotated_trim_bounds_kurt[1]) // 2)

        rotated_trim_bounds_std = (rotated_trim_bounds_std[0] * args.bin, rotated_trim_bounds_std[1] * args.bin)
        rotated_trim_bounds_kurt = (rotated_trim_bounds_kurt[0] * args.bin, rotated_trim_bounds_kurt[1] * args.bin)
        rotated_trim_bounds_combined = (rotated_trim_bounds_combined[0] * args.bin, rotated_trim_bounds_combined[1] * args.bin)
        avg_rotated_trim_bounds = (avg_rotated_trim_bounds[0] * args.bin, avg_rotated_trim_bounds[1] * args.bin)

        save_trim_bounds(avg_rotated_trim_bounds, t, args.path, "trimz_aligned_avg")
        save_trim_bounds(rotated_trim_bounds_combined, t, args.path, "trimz_aligned_combined")
        
        if args.savetrim:
            padded_rotated_trim_bounds = (max(0, avg_rotated_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_rotated_trim_bounds[1] + args.padz))
            padded_rotated_trim_bounds_combined = (max(0, rotated_trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, rotated_trim_bounds_combined[1] + args.padz))

            trimmed_rotated_tomogram_avg = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds)
            trimmed_rotated_tomogram_combined = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds_combined)

            save_trimmed_tomogram(trimmed_rotated_tomogram_avg, t, args.path, suffix="_ali_trimz_avg")
            save_trimmed_tomogram(trimmed_rotated_tomogram_combined, t, args.path, suffix="_ali_trimz_combined")

def fourier_crop(tomogram, bin_factor):
    fft_data = fftshift(fftn(tomogram))
    crop_size = [dim // bin_factor for dim in tomogram.shape]
    cropped_fft_data = fft_data[
        (tomogram.shape[0] // 2 - crop_size[0] // 2):(tomogram.shape[0] // 2 + crop_size[0] // 2),
        (tomogram.shape[1] // 2 - crop_size[1] // 2):(tomogram.shape[1] // 2 + crop_size[1] // 2),
        (tomogram.shape[2] // 2 - crop_size[2] // 2):(tomogram.shape[2] // 2 + crop_size[2] // 2)
    ]
    cropped_fft_data = ifftshift(cropped_fft_data)
    cropped_tomogram = np.real(ifftn(cropped_fft_data))
    return cropped_tomogram

def compute_statistics(tomogram, meanfilter, include_gradient, include_mean, include_skewness, include_entropy):
    means = None
    if include_mean:
        means = np.mean(tomogram, axis=(1, 2))
        if meanfilter > 0:
            nyquist = 0.5
            cutoff = nyquist * meanfilter
            means = gaussian_filter(means, sigma=cutoff)
    
    stds = np.std(tomogram, axis=(1, 2))
    skews = None
    if include_skewness:
        skews = np.zeros(tomogram.shape[0])
        for i in range(tomogram.shape[0]):
            slice_data = tomogram[i].flatten()
            if np.all(slice_data == 0):
                skews[i] = 0
            else:
                skews[i] = np.nan_to_num(skew(slice_data))

    kurt = np.zeros(tomogram.shape[0])
    for i in range(tomogram.shape[0]):
        slice_data = tomogram[i].flatten()
        if np.all(slice_data == 0):
            kurt[i] = 0
        else:
            kurt[i] = np.nan_to_num(kurtosis(slice_data))

    gradients = []
    if include_gradient:
        gradients = [np.mean(sobel(gaussian_filter(tomogram[i], sigma=2))) for i in range(tomogram.shape[0])]
    
    entropies = None
    if include_entropy:
        entropies = []
        for i in range(tomogram.shape[0]):
            try:
                entropy_value = np.mean(entropy(img_as_ubyte(rescale_intensity(tomogram[i], in_range='float', out_range=(0, 1))), disk(5)))
                entropies.append(entropy_value)
            except:
                entropies.append(0)
        entropies = np.array(entropies)
    
    return means, stds, skews, kurt, gradients, entropies

def save_statistics(means, stds, skews, kurt, gradients, entropies, filename, metrics_paths):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    if means is not None:
        np.savetxt(os.path.join(metrics_paths['means'], f'{base}_means.txt'), means)
    np.savetxt(os.path.join(metrics_paths['stds'], f'{base}_stds.txt'), stds)
    if skews is not None:
        np.savetxt(os.path.join(metrics_paths['skews'], f'{base}_skews.txt'), skews)
    np.savetxt(os.path.join(metrics_paths['kurtosis'], f'{base}_kurtosis.txt'), kurt)
    if gradients:
        np.savetxt(os.path.join(metrics_paths['gradients'], f'{base}_gradients.txt'), gradients)
    if entropies is not None:
        np.savetxt(os.path.join(metrics_paths['entropy'], f'{base}_entropy.txt'), entropies)

def plot_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    # Normalize metrics
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')
    
    normalized_metrics = []
    for metric in metrics:
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        normalized_metrics.append(normalized_metric)

    plt.figure()
    for metric, label in zip(normalized_metrics, labels):
        plt.plot(metric, label=label)
    
    # Calculate and plot combined metric
    combined_metric = np.sum(normalized_metrics, axis=0)
    combined_metric = np.nan_to_num(combined_metric)  # Replace nan with zero
    if np.max(combined_metric) != 0:
        combined_metric /= np.max(combined_metric)  # Normalize combined metric
    plt.plot(combined_metric, label='Combined Metric', linestyle='--', color='black')

    plt.xlabel('Z Slice')
    plt.ylabel('Normalized Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.title(f'Statistics for {base}')
    plt.savefig(os.path.join(output_path, f'{base}_statistics.png'), bbox_inches='tight')
    plt.close()

def plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for ax, metric, label in zip(axs, metrics, labels):
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        ax.plot(normalized_metric)
        ax.set_title(label)
        ax.set_xlabel('Z Slice')
        ax.set_ylabel('Normalized Value')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base}_separate_statistics.png'))
    plt.close()

def compute_trim_bounds_from_std(stds):
    gradient_of_stds = np.gradient(stds)
    maxima = np.where((np.r_[True, gradient_of_stds[1:] > gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] > gradient_of_stds[1:], True]))[0]
    minima = np.where((np.r_[True, gradient_of_stds[1:] < gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] < gradient_of_stds[1:], True]))[0]

    # Find the first large maximum and the last large minimum
    start = maxima[0] if len(maxima) > 0 else 0
    end = minima[-1] if len(minima) > 0 else len(stds) - 1

    thickness = end - start
    return (start, end), thickness


def compute_trim_bounds_from_kurtosis(kurt, min_thickness):
    kurt_list = list(kurt)
    max_index = kurt_list.index(max(kurt_list))
    
    # Find the first pseudo-global minimum to the left of the maximum
    kurt_list_left_of_max = kurt_list[:max_index]
    if not kurt_list_left_of_max:
        return (0, len(kurt) - 1), len(kurt) - 1

    left_min_value = min(kurt_list_left_of_max)
    left_min_index = len(kurt_list_left_of_max) - 1 - kurt_list_left_of_max[::-1].index(left_min_value)  # Right-most instance of the minimum value

    # Find the first pseudo-global minimum to the right of the maximum
    kurt_list_right_of_max = kurt_list[max_index:]
    if not kurt_list_right_of_max:
        return (0, len(kurt) - 1), len(kurt) - 1

    right_min_value = min(kurt_list_right_of_max)
    right_min_index = max_index + kurt_list_right_of_max.index(right_min_value)  # Left-most instance of the minimum value

    # Ensure the minima are not too close to each other
    if right_min_index - left_min_index < min_thickness:
        # Adjust the minima indices to satisfy the min_thickness constraint
        kurt_list[left_min_index] = max(kurt_list) + 1  # Exclude the current left minimum
        kurt_list_right_of_max[right_min_index - max_index] = max(kurt_list) + 1  # Exclude the current right minimum
        new_left_min_index = len(kurt_list_left_of_max) - 1 - kurt_list_left_of_max[::-1].index(min(kurt_list_left_of_max))  # Right-most instance
        new_right_min_index = max_index + kurt_list_right_of_max.index(min(kurt_list_right_of_max))  # Left-most instance
        if new_right_min_index - new_left_min_index >= min_thickness:
            left_min_index, right_min_index = new_left_min_index, new_right_min_index

    return (left_min_index, right_min_index), right_min_index - left_min_index


def compute_trim_bounds_combined(stds, kurt, min_thickness):
    combined_metric = stds + kurt
    combined_list = list(combined_metric)
    max_index = combined_list.index(max(combined_list))
    
    # Find the first pseudo-global minimum to the left of the maximum
    combined_list_left_of_max = combined_list[:max_index]
    if not combined_list_left_of_max:
        return (0, len(combined_metric) - 1), len(combined_metric) - 1

    left_min_value = min(combined_list_left_of_max)
    left_min_index = len(combined_list_left_of_max) - 1 - combined_list_left_of_max[::-1].index(left_min_value)  # Right-most instance of the minimum value

    # Find the first pseudo-global minimum to the right of the maximum
    combined_list_right_of_max = combined_list[max_index:]
    if not combined_list_right_of_max:
        return (0, len(combined_metric) - 1), len(combined_metric) - 1

    right_min_value = min(combined_list_right_of_max)
    right_min_index = max_index + combined_list_right_of_max.index(right_min_value)  # Left-most instance of the minimum value

    # Ensure the minima are not too close to each other
    if right_min_index - left_min_index < min_thickness:
        # Adjust the minima indices to satisfy the min_thickness constraint
        combined_list[left_min_index] = max(combined_list) + 1  # Exclude the current left minimum
        combined_list_right_of_max[right_min_index - max_index] = max(combined_list) + 1  # Exclude the current right minimum
        new_left_min_index = len(combined_list_left_of_max) - 1 - combined_list_left_of_max[::-1].index(min(combined_list_left_of_max))  # Right-most instance
        new_right_min_index = max_index + combined_list_right_of_max.index(min(combined_list_right_of_max))  # Left-most instance
        if new_right_min_index - new_left_min_index >= min_thickness:
            left_min_index, right_min_index = new_left_min_index, new_right_min_index

    return (left_min_index, right_min_index), right_min_index - left_min_index


def save_trim_bounds(bounds, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}_{suffix}.txt')
    with open(output_file, 'w') as f:
        f.write(f"{bounds[0]} {bounds[1]}\n")

def save_thicknesses(thickness_std, thickness_kurt, thickness_combined, thickness_avg, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}_thicknesses.txt')
    with open(output_file, 'w') as f:
        f.write(f"STD Thickness: {thickness_std}\n")
        f.write(f"Kurtosis Thickness: {thickness_kurt}\n")
        f.write(f"Combined Thickness: {thickness_combined}\n")
        f.write(f"Average Thickness: {thickness_avg}\n")

def trim_tomogram(tomogram, bounds):
    return tomogram[bounds[0]:bounds[1]]

def save_trimmed_tomogram(trimmed_tomogram, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}{suffix}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(trimmed_tomogram.astype(np.float32))

def align_tomogram_with_pca(tomogram):
    # Downsample the tomogram for alignment
    downsampled = tomogram[::8, ::8, ::8]
    
    # Flatten tomogram data for PCA
    nz, ny, nx = downsampled.shape
    coords = np.array(np.meshgrid(range(nz), range(ny), range(nx))).reshape(3, -1).T
    intensities = downsampled.flatten()
    pca = PCA(n_components=2)
    pca.fit(coords, sample_weight=intensities)
    rotation_matrix = pca.components_
    
    # Rotate the original tomogram
    rotated = apply_rotation(tomogram, rotation_matrix)
    return rotated, rotation_matrix

def apply_rotation(tomogram, rotation_matrix):
    new_shape = tomogram.shape
    new_tomogram = np.zeros(new_shape)
    for z in range(new_shape[0]):
        for y in range(new_shape[1]):
            for x in range(new_shape[2]):
                coords = np.array([z, y, x])
                new_coords = np.dot(rotation_matrix, coords - np.array(new_shape) / 2) + np.array(new_shape) / 2
                if np.all(new_coords >= 0) and np.all(new_coords < new_shape):
                    new_tomogram[z, y, x] = tomogram[tuple(new_coords.astype(int))]
    return new_tomogram

def save_rotated_tomogram(rotated_tomogram, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram.astype(np.float32))

def save_rotation_matrix(rotation_matrix, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}_rotation_matrix.txt')
    np.savetxt(output_file, rotation_matrix)

if __name__ == "__main__":
    main()
'''






'''
#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import mrcfile
from scipy.ndimage import gaussian_filter, sobel
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
import multiprocessing
import psutil

def main():
    print("Starting the script...")
    parser = argparse.ArgumentParser(description="Estimates ice thickness from tomograms")
    parser.add_argument("--align_flat", action="store_true", default=False, help="Align the tomogram to make the region containing the specimen flat")
    parser.add_argument("--bin", type=int, default=1, help="Downsample the tomograms by the given integer factor (default: 1, no downsampling)")
    parser.add_argument("--entropy", action="store_true", default=False, help="Include entropy in the analysis")
    parser.add_argument("--gradient", action="store_true", default=False, help="Include gradient magnitude in the analysis")
    parser.add_argument("--input", type=str, help="Comma-separated .mrc/.hdf/.rec files to process")
    parser.add_argument("--inputdir", type=str, default='.', help="Input directory with multiple .mrc/.hdf/.rec files to process (default: current directory)")
    parser.add_argument("--mean", action="store_true", default=False, help="Include mean density in the analysis")
    parser.add_argument("--meanfilter", type=float, default=0, help="Apply low-pass filter to mean density, expressed as a fraction of Nyquist (0.5 is Nyquist, 1 is half Nyquist, etc.)")
    parser.add_argument("--padz", type=int, default=0, help="Pad the trim boundaries by the given number of pixels (default: 0, no padding)")
    parser.add_argument("--path", type=str, default='tomo_thickness_00', help="Directory to store results")
    parser.add_argument("--savetrim", action="store_true", default=False, help="Save the trimmed versions of the tomograms")
    parser.add_argument("--skewness", action="store_true", default=False, help="Include skewness in the analysis")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for processing (default: 1)")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbose level [0-9]")

    args = parser.parse_args()
    print("Arguments parsed:", args)

    max_threads = multiprocessing.cpu_count()
    if args.threads > max_threads:
        args.threads = max_threads

    available_memory = psutil.virtual_memory().available / (1024**3)  # in GB
    estimated_memory_per_thread = 2  # Estimate 2 GB per thread for safety
    max_threads_based_on_memory = int(available_memory * 0.75 / estimated_memory_per_thread)
    if args.threads > max_threads_based_on_memory:
        args.threads = max_threads_based_on_memory

    if not args.input and not args.inputdir:
        print("\nERROR: Either one of --input or --inputdir required.")
        sys.exit(1)
    
    inputs = []
    if args.inputdir:
        inputs = [os.path.join(args.inputdir, f) for f in os.listdir(args.inputdir) if f.endswith(('.mrc', '.hdf', '.rec'))]
    elif args.input:
        inputs = args.input.split(',')

    for file in inputs:
        if not os.path.isfile(file):
            print(f"ERROR: File not found - {file}")
            sys.exit(1)

    base_path = args.path.rstrip('_00')
    counter = 0
    output_path = f"{base_path}_{counter:02d}"
    while os.path.exists(output_path):
        counter += 1
        output_path = f"{base_path}_{counter:02d}"
    args.path = output_path
    os.makedirs(args.path)

    print("Inputs detected:", inputs)
    
    metrics_paths = {}
    if args.mean:
        metrics_paths['means'] = os.path.join(args.path, 'means')
    metrics_paths['stds'] = os.path.join(args.path, 'stds')
    if args.skewness:
        metrics_paths['skews'] = os.path.join(args.path, 'skews')
    metrics_paths['kurtosis'] = os.path.join(args.path, 'kurtosis')
    if args.gradient:
        metrics_paths['gradients'] = os.path.join(args.path, 'gradients')
    if args.entropy:
        metrics_paths['entropy'] = os.path.join(args.path, 'entropy')

    for path in metrics_paths.values():
        os.makedirs(path, exist_ok=True)

    print("Starting multiprocessing with", args.threads, "threads")
    pool = multiprocessing.Pool(processes=args.threads)
    pool.map(process_tomogram, [(t, args, metrics_paths) for t in inputs])
    pool.close()
    pool.join()

def process_tomogram(params):
    t, args, metrics_paths = params
    print("Processing tomogram:", t)
    
    if not os.path.isfile(t):
        print(f"ERROR: File not found during processing - {t}")
        return
    
    with mrcfile.open(t, permissive=True) as mrc:
        tomogram = mrc.data

    if tomogram is None or tomogram.ndim != 3:
        print(f"ERROR: Tomogram data is invalid or not 3D - {t}")
        return

    if args.bin > 1:
        downsampled_tomogram = fourier_crop(tomogram, args.bin)
    else:
        downsampled_tomogram = tomogram
    
    means, stds, skews, kurt, gradients, entropies = compute_statistics(downsampled_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
    save_statistics(means, stds, skews, kurt, gradients, entropies, t, metrics_paths)
    plot_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    
    trim_bounds_std = compute_trim_bounds_from_std(stds)
    trim_bounds_kurt = compute_trim_bounds_from_kurtosis(kurt)
    trim_bounds_combined = compute_trim_bounds_combined(stds, kurt)

    avg_trim_bounds = ((trim_bounds_std[0] + trim_bounds_kurt[0]) // 2, (trim_bounds_std[1] + trim_bounds_kurt[1]) // 2)

    trim_bounds_std = (trim_bounds_std[0] * args.bin, trim_bounds_std[1] * args.bin)
    trim_bounds_kurt = (trim_bounds_kurt[0] * args.bin, trim_bounds_kurt[1] * args.bin)
    trim_bounds_combined = (trim_bounds_combined[0] * args.bin, trim_bounds_combined[1] * args.bin)
    avg_trim_bounds = (avg_trim_bounds[0] * args.bin, avg_trim_bounds[1] * args.bin)

    save_trim_bounds(avg_trim_bounds, t, args.path, "trimz_avg")
    save_trim_bounds(trim_bounds_combined, t, args.path, "trimz_combined")
    
    if args.savetrim:
        padded_trim_bounds = (max(0, avg_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_trim_bounds[1] + args.padz))
        padded_trim_bounds_combined = (max(0, trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, trim_bounds_combined[1] + args.padz))

        trimmed_tomogram_avg = trim_tomogram(tomogram, padded_trim_bounds)
        trimmed_tomogram_combined = trim_tomogram(tomogram, padded_trim_bounds_combined)

        save_trimmed_tomogram(trimmed_tomogram_avg, t, args.path, suffix="_trimz_avg")
        save_trimmed_tomogram(trimmed_tomogram_combined, t, args.path, suffix="_trimz_combined")

    if args.align_flat:
        rotated_tomogram, rotation_matrix = align_tomogram_with_pca(downsampled_tomogram)
        save_rotated_tomogram(rotated_tomogram, t, args.path)
        save_rotation_matrix(rotation_matrix, t, args.path)
        
        rotated_means, rotated_stds, rotated_skews, rotated_kurt, rotated_gradients, rotated_entropies = compute_statistics(rotated_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
        rotated_trim_bounds_std = compute_trim_bounds_from_std(rotated_stds)
        rotated_trim_bounds_kurt = compute_trim_bounds_from_kurtosis(rotated_kurt)
        rotated_trim_bounds_combined = compute_trim_bounds_combined(rotated_stds, rotated_kurt)

        avg_rotated_trim_bounds = ((rotated_trim_bounds_std[0] + rotated_trim_bounds_kurt[0]) // 2, (rotated_trim_bounds_std[1] + rotated_trim_bounds_kurt[1]) // 2)

        rotated_trim_bounds_std = (rotated_trim_bounds_std[0] * args.bin, rotated_trim_bounds_std[1] * args.bin)
        rotated_trim_bounds_kurt = (rotated_trim_bounds_kurt[0] * args.bin, rotated_trim_bounds_kurt[1] * args.bin)
        rotated_trim_bounds_combined = (rotated_trim_bounds_combined[0] * args.bin, rotated_trim_bounds_combined[1] * args.bin)
        avg_rotated_trim_bounds = (avg_rotated_trim_bounds[0] * args.bin, avg_rotated_trim_bounds[1] * args.bin)

        save_trim_bounds(avg_rotated_trim_bounds, t, args.path, "trimz_aligned_avg")
        save_trim_bounds(rotated_trim_bounds_combined, t, args.path, "trimz_aligned_combined")
        
        if args.savetrim:
            padded_rotated_trim_bounds = (max(0, avg_rotated_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_rotated_trim_bounds[1] + args.padz))
            padded_rotated_trim_bounds_combined = (max(0, rotated_trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, rotated_trim_bounds_combined[1] + args.padz))

            trimmed_rotated_tomogram_avg = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds)
            trimmed_rotated_tomogram_combined = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds_combined)

            save_trimmed_tomogram(trimmed_rotated_tomogram_avg, t, args.path, suffix="_ali_trimz_avg")
            save_trimmed_tomogram(trimmed_rotated_tomogram_combined, t, args.path, suffix="_ali_trimz_combined")

def fourier_crop(tomogram, bin_factor):
    fft_data = fftshift(fftn(tomogram))
    crop_size = [dim // bin_factor for dim in tomogram.shape]
    cropped_fft_data = fft_data[
        (tomogram.shape[0] // 2 - crop_size[0] // 2):(tomogram.shape[0] // 2 + crop_size[0] // 2),
        (tomogram.shape[1] // 2 - crop_size[1] // 2):(tomogram.shape[1] // 2 + crop_size[1] // 2),
        (tomogram.shape[2] // 2 - crop_size[2] // 2):(tomogram.shape[2] // 2 + crop_size[2] // 2)
    ]
    cropped_fft_data = ifftshift(cropped_fft_data)
    cropped_tomogram = np.real(ifftn(cropped_fft_data))
    return cropped_tomogram

def compute_statistics(tomogram, meanfilter, include_gradient, include_mean, include_skewness, include_entropy):
    means = None
    if include_mean:
        means = np.mean(tomogram, axis=(1, 2))
        if meanfilter > 0:
            nyquist = 0.5
            cutoff = nyquist * meanfilter
            means = gaussian_filter(means, sigma=cutoff)
    
    stds = np.std(tomogram, axis=(1, 2))
    skews = None
    if include_skewness:
        skews = np.zeros(tomogram.shape[0])
        for i in range(tomogram.shape[0]):
            slice_data = tomogram[i].flatten()
            if np.all(slice_data == 0):
                skews[i] = 0
            else:
                skews[i] = np.nan_to_num(skew(slice_data))

    kurt = np.zeros(tomogram.shape[0])
    for i in range(tomogram.shape[0]):
        slice_data = tomogram[i].flatten()
        if np.all(slice_data == 0):
            kurt[i] = 0
        else:
            kurt[i] = np.nan_to_num(kurtosis(slice_data))

    gradients = []
    if include_gradient:
        gradients = [np.mean(sobel(gaussian_filter(tomogram[i], sigma=2))) for i in range(tomogram.shape[0])]
    
    entropies = None
    if include_entropy:
        entropies = []
        for i in range(tomogram.shape[0]):
            try:
                entropy_value = np.mean(entropy(img_as_ubyte(rescale_intensity(tomogram[i], in_range='float', out_range=(0, 1))), disk(5)))
                entropies.append(entropy_value)
            except:
                entropies.append(0)
        entropies = np.array(entropies)
    
    return means, stds, skews, kurt, gradients, entropies

def save_statistics(means, stds, skews, kurt, gradients, entropies, filename, metrics_paths):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    if means is not None:
        np.savetxt(os.path.join(metrics_paths['means'], f'{base}_means.txt'), means)
    np.savetxt(os.path.join(metrics_paths['stds'], f'{base}_stds.txt'), stds)
    if skews is not None:
        np.savetxt(os.path.join(metrics_paths['skews'], f'{base}_skews.txt'), skews)
    np.savetxt(os.path.join(metrics_paths['kurtosis'], f'{base}_kurtosis.txt'), kurt)
    if gradients:
        np.savetxt(os.path.join(metrics_paths['gradients'], f'{base}_gradients.txt'), gradients)
    if entropies is not None:
        np.savetxt(os.path.join(metrics_paths['entropy'], f'{base}_entropy.txt'), entropies)

def plot_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    # Normalize metrics
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')
    
    normalized_metrics = []
    for metric in metrics:
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        normalized_metrics.append(normalized_metric)

    plt.figure()
    for metric, label in zip(normalized_metrics, labels):
        plt.plot(metric, label=label)
    
    # Calculate and plot combined metric
    combined_metric = np.sum(normalized_metrics, axis=0)
    combined_metric = np.nan_to_num(combined_metric)  # Replace nan with zero
    if np.max(combined_metric) != 0:
        combined_metric /= np.max(combined_metric)  # Normalize combined metric
    plt.plot(combined_metric, label='Combined Metric', linestyle='--', color='black')

    plt.xlabel('Z Slice')
    plt.ylabel('Normalized Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.title(f'Statistics for {base}')
    plt.savefig(os.path.join(output_path, f'{base}_statistics.png'), bbox_inches='tight')
    plt.close()

def plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for ax, metric, label in zip(axs, metrics, labels):
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        ax.plot(normalized_metric)
        ax.set_title(label)
        ax.set_xlabel('Z Slice')
        ax.set_ylabel('Normalized Value')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base}_separate_statistics.png'))
    plt.close()

def compute_trim_bounds_from_std(stds):
    gradient_of_stds = np.gradient(stds)
    maxima = np.where((np.r_[True, gradient_of_stds[1:] > gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] > gradient_of_stds[1:], True]))[0]
    minima = np.where((np.r_[True, gradient_of_stds[1:] < gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] < gradient_of_stds[1:], True]))[0]

    # Find the first large maximum and the last large minimum
    start = maxima[0] if len(maxima) > 0 else 0
    end = minima[-1] if len(minima) > 0 else len(stds) - 1

    return start, end

def compute_trim_bounds_from_kurtosis(kurt):
    kurt_list = list(kurt)
    first_min_index = kurt_list.index(min(kurt_list))
    kurt_list[first_min_index] = max(kurt_list)
    second_min_index = kurt_list.index(min(kurt_list))

    start, end = sorted([first_min_index, second_min_index])
    return start, end

def compute_trim_bounds_combined(stds, kurt):
    combined_metric = stds + kurt
    gradient_of_combined = np.gradient(combined_metric)
    maxima = np.where((np.r_[True, gradient_of_combined[1:] > gradient_of_combined[:-1]] & np.r_[gradient_of_combined[:-1] > gradient_of_combined[1:], True]))[0]
    minima = np.where((np.r_[True, gradient_of_combined[1:] < gradient_of_combined[:-1]] & np.r_[gradient_of_combined[:-1] < gradient_of_combined[1:], True]))[0]

    # Find the first large maximum and the last large minimum
    start = maxima[0] if len(maxima) > 0 else 0
    end = minima[-1] if len(minima) > 0 else len(combined_metric) - 1

    return start, end

def save_trim_bounds(bounds, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}_{suffix}.txt')
    with open(output_file, 'w') as f:
        f.write(f"{bounds[0]} {bounds[1]}\n")

def trim_tomogram(tomogram, bounds):
    return tomogram[bounds[0]:bounds[1]]

def save_trimmed_tomogram(trimmed_tomogram, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}{suffix}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(trimmed_tomogram.astype(np.float32))

def align_tomogram_with_pca(tomogram):
    # Downsample the tomogram for alignment
    downsampled = tomogram[::8, ::8, ::8]
    
    # Flatten tomogram data for PCA
    nz, ny, nx = downsampled.shape
    coords = np.array(np.meshgrid(range(nz), range(ny), range(nx))).reshape(3, -1).T
    intensities = downsampled.flatten()
    pca = PCA(n_components=2)
    pca.fit(coords, sample_weight=intensities)
    rotation_matrix = pca.components_
    
    # Rotate the original tomogram
    rotated = apply_rotation(tomogram, rotation_matrix)
    return rotated, rotation_matrix

def apply_rotation(tomogram, rotation_matrix):
    new_shape = tomogram.shape
    new_tomogram = np.zeros(new_shape)
    for z in range(new_shape[0]):
        for y in range(new_shape[1]):
            for x in range(new_shape[2]):
                coords = np.array([z, y, x])
                new_coords = np.dot(rotation_matrix, coords - np.array(new_shape) / 2) + np.array(new_shape) / 2
                if np.all(new_coords >= 0) and np.all(new_coords < new_shape):
                    new_tomogram[z, y, x] = tomogram[tuple(new_coords.astype(int))]
    return new_tomogram

def save_rotated_tomogram(rotated_tomogram, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram.astype(np.float32))

def save_rotation_matrix(rotation_matrix, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}_rotation_matrix.txt')
    np.savetxt(output_file, rotation_matrix)

if __name__ == "__main__":
    main()
'''





'''
#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import mrcfile
from scipy.ndimage import gaussian_filter, sobel
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
import multiprocessing
import psutil

def main():
    print("Starting the script...")
    parser = argparse.ArgumentParser(description="Estimates ice thickness from tomograms")
    parser.add_argument("--align_flat", action="store_true", default=False, help="Align the tomogram to make the region containing the specimen flat")
    parser.add_argument("--bin", type=int, default=1, help="Downsample the tomograms by the given integer factor (default: 1, no downsampling)")
    parser.add_argument("--entropy", action="store_true", default=False, help="Include entropy in the analysis")
    parser.add_argument("--gradient", action="store_true", default=False, help="Include gradient magnitude in the analysis")
    parser.add_argument("--input", type=str, help="Comma-separated .mrc/.hdf/.rec files to process")
    parser.add_argument("--inputdir", type=str, default='.', help="Input directory with multiple .mrc/.hdf/.rec files to process (default: current directory)")
    parser.add_argument("--mean", action="store_true", default=False, help="Include mean density in the analysis")
    parser.add_argument("--meanfilter", type=float, default=0, help="Apply low-pass filter to mean density, expressed as a fraction of Nyquist (0.5 is Nyquist, 1 is half Nyquist, etc.)")
    parser.add_argument("--padz", type=int, default=0, help="Pad the trim boundaries by the given number of pixels (default: 0, no padding)")
    parser.add_argument("--path", type=str, default='tomo_thickness_00', help="Directory to store results")
    parser.add_argument("--savetrim", action="store_true", default=False, help="Save the trimmed versions of the tomograms")
    parser.add_argument("--skewness", action="store_true", default=False, help="Include skewness in the analysis")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for processing (default: 1)")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbose level [0-9]")

    args = parser.parse_args()
    print("Arguments parsed:", args)

    max_threads = multiprocessing.cpu_count()
    if args.threads > max_threads:
        args.threads = max_threads

    available_memory = psutil.virtual_memory().available / (1024**3)  # in GB
    estimated_memory_per_thread = 2  # Estimate 2 GB per thread for safety
    max_threads_based_on_memory = int(available_memory * 0.75 / estimated_memory_per_thread)
    if args.threads > max_threads_based_on_memory:
        args.threads = max_threads_based_on_memory

    if not args.input and not args.inputdir:
        print("\nERROR: Either one of --input or --inputdir required.")
        sys.exit(1)
    
    inputs = []
    if args.inputdir:
        inputs = [os.path.join(args.inputdir, f) for f in os.listdir(args.inputdir) if f.endswith(('.mrc', '.hdf', '.rec'))]
    elif args.input:
        inputs = args.input.split(',')

    for file in inputs:
        if not os.path.isfile(file):
            print(f"ERROR: File not found - {file}")
            sys.exit(1)

    base_path = args.path.rstrip('_00')
    counter = 0
    output_path = f"{base_path}_{counter:02d}"
    while os.path.exists(output_path):
        counter += 1
        output_path = f"{base_path}_{counter:02d}"
    args.path = output_path
    os.makedirs(args.path)

    print("Inputs detected:", inputs)
    
    metrics_paths = {
        'means': os.path.join(args.path, 'means'),
        'stds': os.path.join(args.path, 'stds'),
        'skews': os.path.join(args.path, 'skews'),
        'kurtosis': os.path.join(args.path, 'kurtosis'),
        'gradients': os.path.join(args.path, 'gradients'),
        'entropy': os.path.join(args.path, 'entropy'),
    }

    for path in metrics_paths.values():
        os.makedirs(path, exist_ok=True)

    print("Starting multiprocessing with", args.threads, "threads")
    pool = multiprocessing.Pool(processes=args.threads)
    pool.map(process_tomogram, [(t, args, metrics_paths) for t in inputs])
    pool.close()
    pool.join()

def process_tomogram(params):
    t, args, metrics_paths = params
    print("Processing tomogram:", t)
    
    if not os.path.isfile(t):
        print(f"ERROR: File not found during processing - {t}")
        return
    
    with mrcfile.open(t, permissive=True) as mrc:
        tomogram = mrc.data

    if tomogram is None or tomogram.ndim != 3:
        print(f"ERROR: Tomogram data is invalid or not 3D - {t}")
        return

    if args.bin > 1:
        downsampled_tomogram = fourier_crop(tomogram, args.bin)
    else:
        downsampled_tomogram = tomogram
    
    means, stds, skews, kurt, gradients, entropies = compute_statistics(downsampled_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
    save_statistics(means, stds, skews, kurt, gradients, entropies, t, metrics_paths)
    plot_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path, args.mean, args.skewness, args.entropy)
    
    trim_bounds_std = compute_trim_bounds_from_std(stds)
    trim_bounds_kurt = compute_trim_bounds_from_kurtosis(kurt)
    trim_bounds_combined = compute_trim_bounds_combined(stds, kurt)

    avg_trim_bounds = ((trim_bounds_std[0] + trim_bounds_kurt[0]) // 2, (trim_bounds_std[1] + trim_bounds_kurt[1]) // 2)

    trim_bounds_std = (trim_bounds_std[0] * args.bin, trim_bounds_std[1] * args.bin)
    trim_bounds_kurt = (trim_bounds_kurt[0] * args.bin, trim_bounds_kurt[1] * args.bin)
    trim_bounds_combined = (trim_bounds_combined[0] * args.bin, trim_bounds_combined[1] * args.bin)
    avg_trim_bounds = (avg_trim_bounds[0] * args.bin, avg_trim_bounds[1] * args.bin)

    save_trim_bounds(avg_trim_bounds, t, args.path, "trimz_avg.txt")
    save_trim_bounds(trim_bounds_combined, t, args.path, "trimz_combined.txt")
    
    if args.savetrim:
        padded_trim_bounds = (max(0, avg_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_trim_bounds[1] + args.padz))
        padded_trim_bounds_combined = (max(0, trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, trim_bounds_combined[1] + args.padz))

        trimmed_tomogram_avg = trim_tomogram(tomogram, padded_trim_bounds)
        trimmed_tomogram_combined = trim_tomogram(tomogram, padded_trim_bounds_combined)

        save_trimmed_tomogram(trimmed_tomogram_avg, t, args.path, suffix="_trimz_avg")
        save_trimmed_tomogram(trimmed_tomogram_combined, t, args.path, suffix="_trimz_combined")

    if args.align_flat:
        rotated_tomogram, rotation_matrix = align_tomogram_with_pca(downsampled_tomogram)
        save_rotated_tomogram(rotated_tomogram, t, args.path)
        save_rotation_matrix(rotation_matrix, t, args.path)
        
        rotated_means, rotated_stds, rotated_skews, rotated_kurt, rotated_gradients, rotated_entropies = compute_statistics(rotated_tomogram, args.meanfilter, args.gradient, args.mean, args.skewness, args.entropy)
        rotated_trim_bounds_std = compute_trim_bounds_from_std(rotated_stds)
        rotated_trim_bounds_kurt = compute_trim_bounds_from_kurtosis(rotated_kurt)
        rotated_trim_bounds_combined = compute_trim_bounds_combined(rotated_stds, rotated_kurt)

        avg_rotated_trim_bounds = ((rotated_trim_bounds_std[0] + rotated_trim_bounds_kurt[0]) // 2, (rotated_trim_bounds_std[1] + rotated_trim_bounds_kurt[1]) // 2)

        rotated_trim_bounds_std = (rotated_trim_bounds_std[0] * args.bin, rotated_trim_bounds_std[1] * args.bin)
        rotated_trim_bounds_kurt = (rotated_trim_bounds_kurt[0] * args.bin, rotated_trim_bounds_kurt[1] * args.bin)
        rotated_trim_bounds_combined = (rotated_trim_bounds_combined[0] * args.bin, rotated_trim_bounds_combined[1] * args.bin)
        avg_rotated_trim_bounds = (avg_rotated_trim_bounds[0] * args.bin, avg_rotated_trim_bounds[1] * args.bin)

        save_trim_bounds(avg_rotated_trim_bounds, t, args.path, "trimz_aligned_avg.txt")
        save_trim_bounds(rotated_trim_bounds_combined, t, args.path, "trimz_aligned_combined.txt")
        
        if args.savetrim:
            padded_rotated_trim_bounds = (max(0, avg_rotated_trim_bounds[0] - args.padz), min(tomogram.shape[0] - 1, avg_rotated_trim_bounds[1] + args.padz))
            padded_rotated_trim_bounds_combined = (max(0, rotated_trim_bounds_combined[0] - args.padz), min(tomogram.shape[0] - 1, rotated_trim_bounds_combined[1] + args.padz))

            trimmed_rotated_tomogram_avg = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds)
            trimmed_rotated_tomogram_combined = trim_tomogram(rotated_tomogram, padded_rotated_trim_bounds_combined)

            save_trimmed_tomogram(trimmed_rotated_tomogram_avg, t, args.path, suffix="_ali_trimz_avg")
            save_trimmed_tomogram(trimmed_rotated_tomogram_combined, t, args.path, suffix="_ali_trimz_combined")

def fourier_crop(tomogram, bin_factor):
    fft_data = fftshift(fftn(tomogram))
    crop_size = [dim // bin_factor for dim in tomogram.shape]
    cropped_fft_data = fft_data[
        (tomogram.shape[0] // 2 - crop_size[0] // 2):(tomogram.shape[0] // 2 + crop_size[0] // 2),
        (tomogram.shape[1] // 2 - crop_size[1] // 2):(tomogram.shape[1] // 2 + crop_size[1] // 2),
        (tomogram.shape[2] // 2 - crop_size[2] // 2):(tomogram.shape[2] // 2 + crop_size[2] // 2)
    ]
    cropped_fft_data = ifftshift(cropped_fft_data)
    cropped_tomogram = np.real(ifftn(cropped_fft_data))
    return cropped_tomogram

def compute_statistics(tomogram, meanfilter, include_gradient, include_mean, include_skewness, include_entropy):
    means = None
    if include_mean:
        means = np.mean(tomogram, axis=(1, 2))
        if meanfilter > 0:
            nyquist = 0.5
            cutoff = nyquist * meanfilter
            means = gaussian_filter(means, sigma=cutoff)
    
    stds = np.std(tomogram, axis=(1, 2))
    skews = None
    if include_skewness:
        skews = np.zeros(tomogram.shape[0])
        for i in range(tomogram.shape[0]):
            slice_data = tomogram[i].flatten()
            if np.all(slice_data == 0):
                skews[i] = 0
            else:
                skews[i] = np.nan_to_num(skew(slice_data))

    kurt = np.zeros(tomogram.shape[0])
    for i in range(tomogram.shape[0]):
        slice_data = tomogram[i].flatten()
        if np.all(slice_data == 0):
            kurt[i] = 0
        else:
            kurt[i] = np.nan_to_num(kurtosis(slice_data))

    gradients = []
    if include_gradient:
        gradients = [np.mean(sobel(gaussian_filter(tomogram[i], sigma=2))) for i in range(tomogram.shape[0])]
    
    entropies = None
    if include_entropy:
        entropies = []
        for i in range(tomogram.shape[0]):
            try:
                entropy_value = np.mean(entropy(img_as_ubyte(rescale_intensity(tomogram[i], in_range='float', out_range=(0, 1))), disk(5)))
                entropies.append(entropy_value)
            except:
                entropies.append(0)
        entropies = np.array(entropies)
    
    return means, stds, skews, kurt, gradients, entropies

def save_statistics(means, stds, skews, kurt, gradients, entropies, filename, metrics_paths):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    if means is not None:
        np.savetxt(os.path.join(metrics_paths['means'], f'{base}_means.txt'), means)
    np.savetxt(os.path.join(metrics_paths['stds'], f'{base}_stds.txt'), stds)
    if skews is not None:
        np.savetxt(os.path.join(metrics_paths['skews'], f'{base}_skews.txt'), skews)
    np.savetxt(os.path.join(metrics_paths['kurtosis'], f'{base}_kurtosis.txt'), kurt)
    if gradients:
        np.savetxt(os.path.join(metrics_paths['gradients'], f'{base}_gradients.txt'), gradients)
    if entropies is not None:
        np.savetxt(os.path.join(metrics_paths['entropy'], f'{base}_entropy.txt'), entropies)

def plot_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    # Normalize metrics
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')
    
    normalized_metrics = []
    for metric in metrics:
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        normalized_metrics.append(normalized_metric)

    plt.figure()
    for metric, label in zip(normalized_metrics, labels):
        plt.plot(metric, label=label)
    
    # Calculate and plot combined metric
    combined_metric = np.sum(normalized_metrics, axis=0)
    combined_metric = np.nan_to_num(combined_metric)  # Replace nan with zero
    if np.max(combined_metric) != 0:
        combined_metric /= np.max(combined_metric)  # Normalize combined metric
    plt.plot(combined_metric, label='Combined Metric', linestyle='--', color='black')

    plt.xlabel('Z Slice')
    plt.ylabel('Normalized Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.title(f'Statistics for {base}')
    plt.savefig(os.path.join(output_path, f'{base}_statistics.png'), bbox_inches='tight')
    plt.close()

def plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path, include_mean, include_skewness, include_entropy):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    metrics = [stds, kurt]
    labels = ['Standard Deviation', 'Kurtosis']
    if gradients:
        metrics.append(gradients)
        labels.append('Gradient Magnitude')
    if include_mean and means is not None:
        metrics.insert(0, means)
        labels.insert(0, 'Mean Density')
    if include_skewness and skews is not None:
        metrics.insert(-2, skews)
        labels.insert(-2, 'Skewness')
    if include_entropy and entropies is not None:
        metrics.append(entropies)
        labels.append('Entropy')

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for ax, metric, label in zip(axs, metrics, labels):
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        ax.plot(normalized_metric)
        ax.set_title(label)
        ax.set_xlabel('Z Slice')
        ax.set_ylabel('Normalized Value')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base}_separate_statistics.png'))
    plt.close()

def compute_trim_bounds_from_std(stds):
    gradient_of_stds = np.gradient(stds)
    maxima = np.where((np.r_[True, gradient_of_stds[1:] > gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] > gradient_of_stds[1:], True]))[0]
    minima = np.where((np.r_[True, gradient_of_stds[1:] < gradient_of_stds[:-1]] & np.r_[gradient_of_stds[:-1] < gradient_of_stds[1:], True]))[0]

    # Find the first large maximum and the last large minimum
    start = maxima[0] if len(maxima) > 0 else 0
    end = minima[-1] if len(minima) > 0 else len(stds) - 1

    return start, end

def compute_trim_bounds_from_kurtosis(kurt):
    kurt_list = list(kurt)
    first_min_index = kurt_list.index(min(kurt_list))
    kurt_list[first_min_index] = max(kurt_list)
    second_min_index = kurt_list.index(min(kurt_list))

    start, end = sorted([first_min_index, second_min_index])
    return start, end

def compute_trim_bounds_combined(stds, kurt):
    combined_metric = stds + kurt
    gradient_of_combined = np.gradient(combined_metric)
    maxima = np.where((np.r_[True, gradient_of_combined[1:] > gradient_of_combined[:-1]] & np.r_[gradient_of_combined[:-1] > gradient_of_combined[1:], True]))[0]
    minima = np.where((np.r_[True, gradient_of_combined[1:] < gradient_of_combined[:-1]] & np.r_[gradient_of_combined[:-1] < gradient_of_combined[1:], True]))[0]

    # Find the first large maximum and the last large minimum
    start = maxima[0] if len(maxima) > 0 else 0
    end = minima[-1] if len(minima) > 0 else len(combined_metric) - 1

    return start, end

def save_trim_bounds(bounds, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}_{suffix}')
    with open(output_file, 'w') as f:
        f.write(f"{bounds[0]} {bounds[1]}\n")

def trim_tomogram(tomogram, bounds):
    return tomogram[bounds[0]:bounds[1]]

def save_trimmed_tomogram(trimmed_tomogram, filename, output_path, suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    output_file = os.path.join(output_path, f'{base}{suffix}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(trimmed_tomogram.astype(np.float32))

def align_tomogram_with_pca(tomogram):
    # Downsample the tomogram for alignment
    downsampled = tomogram[::8, ::8, ::8]
    
    # Flatten tomogram data for PCA
    nz, ny, nx = downsampled.shape
    coords = np.array(np.meshgrid(range(nz), range(ny), range(nx))).reshape(3, -1).T
    intensities = downsampled.flatten()
    pca = PCA(n_components=2)
    pca.fit(coords, sample_weight=intensities)
    rotation_matrix = pca.components_
    
    # Rotate the original tomogram
    rotated = apply_rotation(tomogram, rotation_matrix)
    return rotated, rotation_matrix

def apply_rotation(tomogram, rotation_matrix):
    new_shape = tomogram.shape
    new_tomogram = np.zeros(new_shape)
    for z in range(new_shape[0]):
        for y in range(new_shape[1]):
            for x in range(new_shape[2]):
                coords = np.array([z, y, x])
                new_coords = np.dot(rotation_matrix, coords - np.array(new_shape) / 2) + np.array(new_shape) / 2
                if np.all(new_coords >= 0) and np.all(new_coords < new_shape):
                    new_tomogram[z, y, x] = tomogram[tuple(new_coords.astype(int))]
    return new_tomogram

def save_rotated_tomogram(rotated_tomogram, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram.astype(np.float32))

def save_rotation_matrix(rotation_matrix, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_ali')
    output_file = os.path.join(output_path, f'{base}_rotation_matrix.txt')
    np.savetxt(output_file, rotation_matrix)

if __name__ == "__main__":
    main()
'''















'''
#!/usr/bin/env python

# Author: Jesus Galaz-Montoya 04/2020; last modification: 07/2024, updated by ChatGPT 07/2024

import os
import sys
import numpy as np
import argparse
import mrcfile
from scipy.ndimage import gaussian_filter, sobel, affine_transform
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
import multiprocessing
import psutil

def main():
    print("Starting the script...")
    parser = argparse.ArgumentParser(description="Estimates ice thickness from tomograms")
    parser.add_argument("--ali2slab", action="store_true", default=False, help="Make a gradient slab to try to make the tomogram contents 'flat'")
    parser.add_argument("--inputdir", type=str, default='.', help="Input directory with multiple .mrc/.hdf/.rec files to process (default: current directory)")
    parser.add_argument("--input", type=str, help="Comma-separated .mrc/.hdf/.rec files to process")
    parser.add_argument("--path", type=str, default='tomo_thickness_00', help="Directory to store results")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbose level [0-9]")
    parser.add_argument("--rotate", action="store_true", default=False, help="Rotate the tomogram to make the region containing the specimen flat")
    parser.add_argument("--savetrim", action="store_true", default=False, help="Save the trimmed versions of the tomograms")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for processing (default: 1)")

    args = parser.parse_args()
    print("Arguments parsed:", args)

    # Limit threads to the number of available CPUs
    max_threads = multiprocessing.cpu_count()
    if args.threads > max_threads:
        args.threads = max_threads

    # Adjust threads based on memory usage
    available_memory = psutil.virtual_memory().available / (1024**3)  # in GB
    estimated_memory_per_thread = 2  # Estimate 2 GB per thread for safety
    max_threads_based_on_memory = int(available_memory * 0.75 / estimated_memory_per_thread)
    if args.threads > max_threads_based_on_memory:
        args.threads = max_threads_based_on_memory

    if not args.input and not args.inputdir:
        print("\nERROR: Either one of --input or --inputdir required.")
        sys.exit(1)
    
    inputs = []
    if args.inputdir:
        inputs = [os.path.join(args.inputdir, f) for f in os.listdir(args.inputdir) if f.endswith(('.mrc', '.hdf', '.rec'))]
    elif args.input:
        inputs = args.input.split(',')

    # Check if all files exist
    for file in inputs:
        if not os.path.isfile(file):
            print(f"ERROR: File not found - {file}")
            sys.exit(1)

    # Ensure the output directory is uniquely numbered
    base_path = args.path.rstrip('_00')
    counter = 0
    output_path = f"{base_path}_{counter:02d}"
    while os.path.exists(output_path):
        counter += 1
        output_path = f"{base_path}_{counter:02d}"
    args.path = output_path
    os.makedirs(args.path)

    print("Inputs detected:", inputs)
    
    # Create output directories
    metrics_paths = {
        'means': os.path.join(args.path, 'means'),
        'stds': os.path.join(args.path, 'stds'),
        'skews': os.path.join(args.path, 'skews'),
        'kurtosis': os.path.join(args.path, 'kurtosis'),
        'gradients': os.path.join(args.path, 'gradients'),
        'entropy': os.path.join(args.path, 'entropy'),
    }

    for path in metrics_paths.values():
        os.makedirs(path, exist_ok=True)

    # Use multiprocessing to handle multiple tomograms
    print("Starting multiprocessing with", args.threads, "threads")
    pool = multiprocessing.Pool(processes=args.threads)
    pool.map(process_tomogram, [(t, args, metrics_paths) for t in inputs])
    pool.close()
    pool.join()

def process_tomogram(params):
    t, args, metrics_paths = params
    print("Processing tomogram:", t)
    
    if not os.path.isfile(t):
        print(f"ERROR: File not found during processing - {t}")
        return
    
    with mrcfile.open(t, permissive=True) as mrc:
        tomogram = mrc.data
    
    means, stds, skews, kurt, gradients, entropies = compute_statistics(tomogram)
    save_statistics(means, stds, skews, kurt, gradients, entropies, t, metrics_paths)
    plot_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path)
    plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, t, args.path)
    
    trim_bounds = compute_trim_bounds(means, stds, kurt, gradients, entropies)
    save_trim_bounds(trim_bounds, t, args.path, "trimz.txt")
    
    if args.savetrim:
        trimmed_tomogram = trim_tomogram(tomogram, trim_bounds)
        save_trimmed_tomogram(trimmed_tomogram, t, args.path)

    if args.rotate:
        rotated_tomogram, rotation_matrix = align_tomogram_with_pca(tomogram)
        save_rotated_tomogram(rotated_tomogram, t, args.path)
        save_rotation_matrix(rotation_matrix, t, args.path)
        
        rotated_means, rotated_stds, rotated_skews, rotated_kurt, rotated_gradients, rotated_entropies = compute_statistics(rotated_tomogram)
        rotated_trim_bounds = compute_trim_bounds(rotated_means, rotated_stds, rotated_kurt, rotated_gradients, rotated_entropies)
        save_trim_bounds(rotated_trim_bounds, t, args.path, "trimz_aligned.txt")
        
        if args.savetrim:
            trimmed_rotated_tomogram = trim_tomogram(rotated_tomogram, rotated_trim_bounds)
            save_trimmed_tomogram(trimmed_rotated_tomogram, t, args.path, rotated=True)

def compute_statistics(tomogram):
    means = np.mean(tomogram, axis=(1, 2))
    stds = np.std(tomogram, axis=(1, 2))
    skews = skew(tomogram, axis=(1, 2))
    kurt = kurtosis(tomogram, axis=(1, 2))
    gradients = [np.mean(sobel(gaussian_filter(tomogram[i], sigma=2))) for i in range(tomogram.shape[0])]
    
    # Convert each slice to uint8 for entropy calculation
    entropies = [np.mean(entropy(img_as_ubyte(rescale_intensity(tomogram[i], in_range='float', out_range=(0, 1))), disk(5))) for i in range(tomogram.shape[0])]
    
    return means, stds, skews, kurt, gradients, entropies

def save_statistics(means, stds, skews, kurt, gradients, entropies, filename, metrics_paths):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    np.savetxt(os.path.join(metrics_paths['means'], f'{base}_means.txt'), means)
    np.savetxt(os.path.join(metrics_paths['stds'], f'{base}_stds.txt'), stds)
    np.savetxt(os.path.join(metrics_paths['skews'], f'{base}_skews.txt'), skews)
    np.savetxt(os.path.join(metrics_paths['kurtosis'], f'{base}_kurtosis.txt'), kurt)
    np.savetxt(os.path.join(metrics_paths['gradients'], f'{base}_gradients.txt'), gradients)
    np.savetxt(os.path.join(metrics_paths['entropy'], f'{base}_entropy.txt'), entropies)

def plot_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    # Normalize metrics
    metrics = [means, stds, skews, kurt, gradients, entropies]
    normalized_metrics = []
    for metric in metrics:
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        normalized_metrics.append(normalized_metric)

    plt.figure()
    labels = ['Mean Density', 'Standard Deviation', 'Skewness', 'Kurtosis', 'Gradient Magnitude', 'Entropy']
    for metric, label in zip(normalized_metrics, labels):
        plt.plot(metric, label=label)
    
    # Calculate and plot combined metric
    combined_metric = np.sum(normalized_metrics, axis=0)
    plt.plot(combined_metric, label='Combined Metric', linestyle='--', color='black')

    plt.xlabel('Z Slice')
    plt.ylabel('Normalized Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.title(f'Statistics for {base}')
    plt.savefig(os.path.join(output_path, f'{base}_statistics.png'), bbox_inches='tight')
    plt.close()

def plot_separate_statistics(means, stds, skews, kurt, gradients, entropies, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    
    metrics = [means, stds, skews, kurt, gradients, entropies]
    labels = ['Mean Density', 'Standard Deviation', 'Skewness', 'Kurtosis', 'Gradient Magnitude', 'Entropy']

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for ax, metric, label in zip(axs, metrics, labels):
        if np.max(metric) != np.min(metric):
            normalized_metric = (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            normalized_metric = metric
        ax.plot(normalized_metric)
        ax.set_title(label)
        ax.set_xlabel('Z Slice')
        ax.set_ylabel('Normalized Value')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base}_separate_statistics.png'))
    plt.close()

def compute_trim_bounds(means, stds, kurt, gradients, entropies, threshold=0.1):
    # Normalize the metrics
    def normalize(metric):
        if np.max(metric) != np.min(metric):
            return (metric - np.min(metric)) / (np.max(metric) - np.min(metric))
        else:
            return metric
    
    stds = normalize(stds)
    kurt = normalize(kurt)
    gradients = normalize(gradients)
    entropies = normalize(entropies)

    # Combine metrics
    combined_metric = stds + kurt + gradients + entropies
    combined_metric /= np.max(combined_metric)

    mask = combined_metric > threshold
    z_indices = np.where(mask)[0]
    if z_indices.size == 0:
        return 0, len(means) - 1
    return z_indices[0], z_indices[-1]

def save_trim_bounds(trim_bounds, filename, output_path, file_suffix):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '')
    trim_file = os.path.join(output_path, f'{base}_{file_suffix}')
    with open(trim_file, 'w') as f:
        f.write(f'{trim_bounds[0]} {trim_bounds[1]}\n')

def trim_tomogram(tomogram, trim_bounds):
    return tomogram[trim_bounds[0]:trim_bounds[1] + 1]

def save_trimmed_tomogram(trimmed_tomogram, filename, output_path, rotated=False):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_trimmed')
    if rotated:
        base += '_rotated'
    output_file = os.path.join(output_path, f'{base}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(trimmed_tomogram.astype(np.float32))

def align_tomogram_with_pca(tomogram):
    # Downsample the tomogram for alignment
    downsampled = tomogram[::8, ::8, ::8]
    
    # Flatten tomogram data for PCA
    nz, ny, nx = downsampled.shape
    coords = np.array(np.meshgrid(range(nz), range(ny), range(nx), indexing='ij'))
    coords = coords.reshape(3, -1).T
    values = downsampled.flatten()
    
    # Filter out low-intensity voxels
    mask = values > np.percentile(values, 90)
    coords = coords[mask]
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(coords)
    principal_axes = pca.components_
    
    # Compute rotation matrix
    rotation_matrix = np.linalg.inv(principal_axes)
    
    # Apply rotation to tomogram
    rotated = apply_rotation(tomogram, rotation_matrix)
    
    return rotated, rotation_matrix

def apply_rotation(tomogram, rotation_matrix):
    # Apply the affine transformation to the tomogram
    rotated_tomogram = affine_transform(tomogram, rotation_matrix, order=1, mode='constant')

    return rotated_tomogram

def save_rotated_tomogram(rotated_tomogram, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_rotated')
    output_file = os.path.join(output_path, f'{base}.mrc')
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram.astype(np.float32))

def save_rotation_matrix(rotation_matrix, filename, output_path):
    base = os.path.basename(filename).replace('.mrc', '').replace('.hdf', '').replace('.rec', '_rotation_matrix')
    output_file = os.path.join(output_path, f'{base}.txt')
    np.savetxt(output_file, rotation_matrix)

if __name__ == "__main__":
    main()
    sys.stdout.flush()
'''



''' #OLD VERSION
from __future__ import print_function
from __future__ import division
from past.utils import old_div
from builtins import range

import math
import numpy as np
import random
import os
import sys

from EMAN2 import *
from EMAN2_utils import *

def main():
    progname = os.path.basename(sys.argv[0])
    usage = """prog [options]

    Estimates ice thickness from tomograms
    """
            
    parser = EMArgumentParser(usage=usage,version=EMANVERSION)
    
    parser.add_argument("--ali2slab", action="store_true", default=False, help="""Default=False. Make a gradient slab to try to make the tomgoram contents 'flat'""")
    parser.add_argument("--align",type=str,default="rotate_translate_3d_tree",help="""Default is rotate_translate_3d_tree. See e2help.py aligners to see the list of parameters the aligner takes.""")
    parser.add_argument("--aligncmp",type=str,default="ccc.tomo.thresh",help="""Default=ccc.tomo.thresh. The comparator used for the --align aligner. Do not specify unless you need to use another specific aligner.""")
    
    parser.add_argument("--inputdir", type=str, default=None, help="""Default=None. Input directory with multiple .json files to process, usually an eman2 'info' directory, or a directory containing subtomogram stacks.""")

    parser.add_argument("--input", type=str, default=None, help="""Default=None. Comma-separated .json files to extract coordinates from.""")
    
    #parser.add_argument("--lossoutput", type=str, default=None, help="""Default=None. txt file to write output of losses analysis for a set of tomograms to.""")
    #parser.add_argument("--loss", action="store_true", default=False, help="""Default=False. Extract 'loss' or alingment error for tomograms from .json file.""")

    parser.add_argument("--path",type=str,default='icethickness',help="""Defautl=icethickness. Directory to store results in. The default is a numbered series of directories containing the prefix 'icethickness'; for example, icethickness_02 will be the directory by default if 'icethickness_01' already exists.""")
    parser.add_argument("--ppid", type=int, help="""Default=-1. Set the PID of the parent process, used for cross platform PPID""",default=-1)

    #parser.add_argument("--tomobin", type=int, default=4, help="""Default=4. Binning of the tomogram in which the particles were picked""")
    #parser.add_argument("--tomosize", type=str, default="1024,1024,256", help="""Default=1024,1024,256. X, Y, Z dimensions of the tomogram in which particles were picked""")   

    parser.add_argument("--verbose", "-v", type=int, default=0, help="""Default=0. Verbose level [0-9], higher number means higher level of verboseness.""")

    (options, args) = parser.parse_args()
    

    if not options.input and not options.inputdir:
        print("\nERROR: Either one of --input or --inputdir required.")
        sys.exit(1)
    #if not options.output:
    #   options.output = os.path.basename(os.path.splittext(options.input))[0]+'.txt'
        #print("\nERROR: --output required")
        #sys.exit(1)

    if options.inputdir and options.input:
        print("\nERROR: --input and --inputdir can not be provided at the same time")
        sys.exit(1)

    inputs=[]
    if options.inputdir:
        try:
            inputs = [options.inputdir+'/'+t for t in os.listdir(options.inputdir) if '.hdf' in t[-4:] and "preproc" in t and EMData(t,0,True)]
            n=len(inputs)
            print("\nfound these many inputs n={}, which are {}".format(n,inputs))
        except:
            print("\nERROR: something wrong with --inputdir={} or the files in it {}".format(options.inputdir, os.getcwd(options.inputdir)))
            sys.exit(1)
    elif options.input:
        inputs.append(options.input)

    inputs.sort()

    options = makepath(options,'icethickness')
    writeParameters( options, 'measure_ice.py', 'icethickness' )
    
    slabfile=None
    slabimg=None
    if options.ali2slab:
        options = sptOptionsParser( options )
        slabfile,slabimg=makeslab(options,inputs)

    cmds=['e2proc2d.py ' + t + ' ' + t.replace('.hdf','_2d.hdf') + ' --threed2twod' for t in inputs]
    
    for cmd in cmds:
        runcmd(options,cmd)

    os.mkdir(options.path+'/means')
    os.mkdir(options.path+'/stds')

    imgsali=[]
    for t in inputs:
        procimg(options,t)

        if options.ali2slab and slabimg and slabfile:
            simage = EMData(t,0)
            aliparameters = simage.xform_align_nbest("rotate_translate_3d_tree",slabimg,{"sym":"d1"},1)
            T=aliparameters[0]['xform.align3d']
            simage_ali=simage.copy()
            simage_ali.transform(T)

            imgalifile=options.path+'/'+t.replace(".hdf","_ali.hdf")
            imgsali.append(imgalifile)
            simage_ali.write_image(imgalifile,0)


    cmdplot = "cd " + options.path + " && e2plotfig_jan_2021.py --datay "
    for f in os.listdir(options.path+'/means').sort():
        if 'meansz.txt' in f:
            cmdplot += 'means/'+ f +','
    cmdplot.rstrip(',')

    #cmdplot += " --datay_err "
    #for f in os.listdir(options.path+'/stds').sort():
    #    if 'stdsz.txt' in f:
    #        cmdplot += 'stds/'+ f +','
    #cmdplot.rstrip(',')

    cmdplot += " --labelxaxis z_slice --labelyaxis mean_density --unitsy AU --mult 10 --unitsx N --scaleaxes --highresolution --labeltitle ice_thickness"
    runcmd(options,cmdplot)


    if options.ali2slab and slabimg and slabfile:
        os.mkdir(options.path+'/means_ali')
        os.mkdir(options.path+'/stds_ali')

        cmdsali=['e2proc2d.py ' + t + ' ' + t.replace('.hdf','_2d.hdf') + ' --threed2twod' for t in imgsali]
        for cmd in cmds:
            runcmd(options,cmd)

        if imgsali:
            for t in imgsali:
                procimg(options,t,'_ali')

            cmdplot_ali = "cd " + options.path + " && e2plotfig_jan_2021.py --datay "
            for f in os.listdir(options.path+'/means_ali').sort():
                if 'meansz_ali.txt' in f:
                    cmdplot_ali += 'means_ali/'+ f +','
            cmdplot_ali.rstrip(',')

            #cmdplot_ali += " --datay_err "
            #for f in os.listdir(options.path+'/stds_ali').sort():
            #    if 'stdsz_ali.txt' in f:
            #       cmdplot_ali += 'stds_ali/'+ f +','
            #cmdplot_ali.rstrip(',')

            cmdplot_ali += " --labelxaxis z_slice --labelyaxis mean_density --unitsy AU --mult 10 --unitsx N --scaleaxes --highresolution --labeltitle ice_thickness"
            runcmd(options,cmdplot_ali)

    E2end(logger)

    return

def procimg(options,t,tag=''):
    input2d = t.replace('.hdf','_2d.hdf')
    n = EMUtil.get_image_count(input2d)
    hdr = EMData(input2d,0,True)
    nz=hdr['nz']
    if n!= nz:
        print("\nWEIRD: for f={} nz={} != n={}".format(input2d,nz,n))
        sys.exit(1)

    linesm = [ str(EMData(input2d,i,True)['mean'])+"\n" for i in range(n) ]
    liness = [ str(EMData(input2d,i,True)['sigma'])+"\n" for i in range(n) ]

    meanfile = options.path + '/means'+tag+'/' + t.replace('.hdf','_meansz'+tag+'.txt') 
    stdfile = options.path + '/stds'+tag+'/' + t.replace('.hdf','_stdsz'+tag+'.txt') 
    with open(meanfile,'w') as f:
        f.writelines(linesm)
    with open(stdfile,'w') as g:
        g.writelines(liness)
    return


def makeslab(options,inputs):
    hdr=EMData(inputs[0],0,True)
    nx=hdr['nx']
    ny=hdr['ny']
    nz=hdr['nz']

    a=EMData(nx,ny,nz)
    a.to_one()

    thickz=nz/4.0
    maskz=int(nz/2.0-thickz/2.0)
    am=a.process("mask.zeroedge3d",{"z0":maskz,"z1":maskz})
    amlp=am.process("filter.lowpass.gauss",{"cutoff_freq":0.005})
    slabfile=options.path+"/slab_z"+str(int(trickz))+"lp200.hdf"
    amlp.write_image(slabfile,0)
    return slabfile,amlp



if __name__ == "__main__":
    main()
    sys.stdout.flush()
'''