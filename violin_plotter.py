#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from itertools import combinations

def load_data_file(file_path):
    """
    Load data from a single-column text file.
    The file should contain one value per line.
    """
    values = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Skip header lines if they can't be converted to float
                    val = float(line.strip())
                    values.append(val)
                except ValueError:
                    continue
        return np.array(values)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return np.array([])

def remove_low_outliers(data, method='percentile', threshold=0.0):
    """
    Remove low-value outliers from a data array.
    
    Parameters:
        data: NumPy array of values
        method: Method to use for outlier detection ('percentile', 'zscore', or 'iqr')
        threshold: Threshold for outlier detection:
                   - For percentiles: values between 0-1 (e.g., 0.05 for bottom 5%)
                   - For z-score: values > 1 (e.g., 2.0 for 2 standard deviations)
                   - For IQR: typically 1.5 (points below Q1 - 1.5*IQR are outliers)
                   
    Returns:
        NumPy array with outliers removed
    """
    if not isinstance(data, np.ndarray) or len(data) == 0 or threshold <= 0:
        return data  # No outlier removal
        
    if method == 'percentile':
        # Remove bottom N% of points
        if 0 < threshold < 1:
            cutoff = np.percentile(data, threshold * 100)
            print(f"Removing {threshold*100:.1f}% lowest values (< {cutoff:.4f})")
        else:
            return data  # Invalid percentile
            
    elif method == 'zscore':
        # Remove points below Z standard deviations from the mean
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            cutoff = mean_val - threshold * std_val
            print(f"Removing values with z-score < -{threshold:.1f} (< {cutoff:.4f})")
        else:
            return data  # Can't compute z-score with std=0
            
    elif method == 'iqr':
        # Remove points below Q1 - threshold*IQR
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        if iqr > 0:
            cutoff = q1 - threshold * iqr
            print(f"Removing values below Q1-{threshold:.1f}*IQR (< {cutoff:.4f})")
        else:
            return data  # Can't use IQR when it's 0
    else:
        return data  # Unknown method
        
    # Filter data
    filtered_data = data[data >= cutoff]
    removed_count = len(data) - len(filtered_data)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers ({removed_count/len(data)*100:.1f}% of values)")
        
    return filtered_data

def compute_statistics(data1, data2):
    """
    Compute statistical significance and effect size between two distributions.
    
    Returns:
        p-value, effect size, method used, and significance stars
    """
    def significance_from_pval(p):
        # More granular significance levels with no artificial cap
        if p < 0.00001: return "*****"  # 5 stars for extremely significant
        elif p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "ns"
        
    def is_normal(dat):
        if len(dat) < 3:
            return False
        try:
            stat, p = shapiro(dat)
            return p > 0.05
        except:
            return False
            
    # Calculate effect size and p-value
    normal1 = is_normal(data1)
    normal2 = is_normal(data2)
    
    if normal1 and normal2:
        # For normally distributed data, use t-test and Cohen's d
        tstat, pval = ttest_ind(data1, data2, equal_var=False)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        effect = 0.0 if pooled_std < 1e-12 else abs(mean1 - mean2) / pooled_std
        method = "Cohen's d"
    else:
        # For non-normal data, use Mann-Whitney U test and rank-biserial correlation
        try:
            stat, pval = mannwhitneyu(data1, data2, alternative='two-sided')
            n1, n2 = len(data1), len(data2)
            effect = 1 - (2*stat)/(n1*n2)  # rank-biserial correlation
            method = "Rank-biserial"
        except ValueError:
            # Handle case where Mann-Whitney U test fails (e.g., identical arrays)
            return 1.0, 0.0, "N/A", "ns"
            
    sig = significance_from_pval(pval)
    return pval, effect, method, sig

def plot_violins(data_files, labels=None, drop_outliers=0.0, outlier_method="percentile", 
              output="comparative_violin_plots.png", show_pvalues=False, summary_table=False, title="Distribution Comparison"):
    """
    Create violin plots for multiple datasets and calculate pairwise statistics.
    
    Parameters (in alphabetical order):
        data_files: List of paths to data files
        drop_outliers: Threshold for removing low-value outliers
        labels: Labels for each dataset
        outlier_method: Method for outlier detection ('percentile', 'zscore', or 'iqr')
        output: Path to save the output plot (default: "comparative_violin_plots.png")
        show_pvalues: Whether to show p-values in addition to effect sizes
        summary_table: Whether to include a summary table of statistics below the plot
        title: Title for the plot
    """
    # Load data
    datasets = []
    for file_path in data_files:
        data = load_data_file(file_path)
        if len(data) > 0:
            # Apply outlier removal if requested
            if drop_outliers > 0:
                data = remove_low_outliers(data, method=outlier_method, threshold=drop_outliers)
            datasets.append(data)
        else:
            print(f"Warning: No valid data in {file_path}, skipping")
    
    if len(datasets) == 0:
        print("Error: No valid datasets to plot")
        return
    
    # Default labels if not provided
    if not labels or len(labels) != len(datasets):
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]
    
    # Create figure
    fig_width = max(8, len(datasets) * 1.5)  # Scale width based on dataset count
    fig_height = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot violins
    positions = list(range(1, len(datasets) + 1))
    parts = ax.violinplot(datasets, positions=positions, showmeans=False, 
                          showmedians=False, showextrema=False)
    
    # Customize violins
    colors = plt.cm.tab10.colors[:len(datasets)]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Add data points, medians, and means
    for i, (data, pos) in enumerate(zip(datasets, positions)):
        color = colors[i % len(colors)]
        
        # Add scatter plot with jitter
        jitter = 0.1
        x = np.full(data.shape, pos) + np.random.uniform(-jitter, jitter, size=len(data))
        ax.scatter(x, data, s=5, alpha=0.5, color=color, edgecolor='none')
        
        # Add median and mean markers
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        mean = np.mean(data)
        
        ax.scatter(pos, median, marker='s', color='red', s=30, zorder=3)
        ax.scatter(pos, mean, marker='o', color='white', edgecolor='black', s=30, zorder=3)
        
        # Add vertical lines showing data range and IQR
        ax.vlines(pos, data.min(), data.max(), color='black', lw=1)
        ax.vlines(pos, q1, q3, color='black', lw=3)
    
    # Add statistics
    if len(datasets) > 1:
        # Calculate all pairwise stats
        y_max = max([d.max() for d in datasets])
        y_min = min([d.min() for d in datasets])
        y_range = y_max - y_min
        
        stats_text = []
        
        # Compute and add pairwise statistics
        pairs = list(combinations(range(len(datasets)), 2))
        line_height = y_range * 0.05  # Space between significance lines
        
        for idx, (i, j) in enumerate(pairs):
            # Calculate statistics
            pval, effect, method, sig = compute_statistics(datasets[i], datasets[j])
            
            # Determine y position for significance bar
            y_pos = y_max + (idx + 1) * line_height
            
            # Draw significance bar
            x1, x2 = positions[i], positions[j]
            bar_height = 0.02 * y_range
            
            # Draw the bar
            ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], 
                    lw=1.5, c='black')
            
            # Add annotation (p-value and/or effect size)
            if show_pvalues:
                # Show both p-value and effect size
                ax.text((x1 + x2) / 2, y_pos + bar_height, f"{sig}, d={effect:.2f}", 
                        ha='center', va='bottom', fontsize=10)
            else:
                # Show only effect size
                ax.text((x1 + x2) / 2, y_pos + bar_height, f"d={effect:.2f}", 
                        ha='center', va='bottom', fontsize=10)
            
            # Add to stats text for the table
            if show_pvalues:
                stats_text.append(f"{labels[i]} vs {labels[j]}: p={pval:.4f} ({sig}), {method}={effect:.3f}")
            else:
                stats_text.append(f"{labels[i]} vs {labels[j]}: {method}={effect:.3f}")
        
        # Set y-axis limit to include significance bars
        if pairs:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + (len(pairs) + 1) * line_height)
        
        # Optionally add a summary table as text annotation
        if stats_text and summary_table:
            # Add a text summary at the bottom of the plot (simpler approach)
            summary = "\n".join(stats_text)
            # First ensure tight layout for the main plot
            plt.tight_layout()
            # Then add text below the plot
            plt.figtext(0.5, 0.01, summary, 
                      fontsize=9, va="bottom", ha="center",
                      bbox={"boxstyle": "round", "alpha": 0.1, "pad": 0.5},
                      wrap=True)
    
    # Customize plot appearance
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Values")
    
    # Create more descriptive title
    dataset_counts = ", ".join([f"{label}: {len(data)}" for label, data in zip(labels, datasets)])
    ax.set_title(f"{title}\n{dataset_counts}")
    
    plt.tight_layout()
    
    # Save or show
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Create violin plots comparing multiple distributions and calculate statistical significance.")
    
    # Arguments in alphabetical order
    parser.add_argument("--data_labels", default=None,
                        help="Comma-separated labels for each dataset (must match number of files).")
    parser.add_argument("--display_only", action='store_true', default=False,
                        help="Only display the plot without saving it. Default=False (plot is saved).")
    parser.add_argument("--drop_outliers", type=float, default=0.0,
                        help="Drop low-value outliers based on percentile or z-score. Values between 0-1 are treated as percentiles (e.g., 0.05 drops bottom 5%%). Values >1 are treated as z-score thresholds. Default=0.0 (no outlier removal).")
    parser.add_argument("--files", required=True, 
                        help="Comma-separated list of data files. Each file should contain one value per line.")
    parser.add_argument("--outlier_method", type=str, choices=['percentile', 'zscore', 'iqr'], default='percentile',
                        help="Method for detecting outliers: 'percentile', 'zscore', or 'iqr' (Interquartile Range). Default='percentile'.")
    parser.add_argument("--output", default="comparative_violin_plots.png",
                        help="Output file path. Default='comparative_violin_plots.png'.")
    parser.add_argument("--show_pvalues", action='store_true', default=False,
                        help="Show p-values in addition to effect sizes. By default, only effect sizes are shown.")
    parser.add_argument("--summary_table", action='store_true', default=False,
                        help="Include a summary table of statistics below the plot. Default=False.")
    parser.add_argument("--title", default="Distribution Comparison",
                        help="Title for the plot.")
    
    args = parser.parse_args()
    
    # Parse input files
    data_files = [f.strip() for f in args.files.split(",")]
    
    # Parse labels if provided
    labels = None
    if args.data_labels:
        labels = [l.strip() for l in args.data_labels.split(",")]
        
    # Validate files
    valid_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not valid_files:
        print("Error: No valid files to process")
        sys.exit(1)
        
    # Output setting
    output_path = None if args.display_only else args.output
        
    # Create plot with parameters in alphabetical order
    plot_violins(
        data_files=valid_files,
        drop_outliers=args.drop_outliers,
        labels=labels,
        outlier_method=args.outlier_method,
        output=output_path,
        show_pvalues=args.show_pvalues,
        summary_table=args.summary_table,
        title=args.title
    )

if __name__ == "__main__":
    main()