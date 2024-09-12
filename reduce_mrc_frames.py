import numpy as np
import mrcfile
import argparse

def read_mrc_file(mrc_filename):
    """
    Reads the MRC file and returns the data as a 3D numpy array.
    
    Parameters:
        mrc_filename (str): Path to the MRC file.
    
    Returns:
        numpy.ndarray: The 3D array of frames from the MRC file.
    """
    
    print(f'\nreading {mrc_filename}')
    with mrcfile.open(mrc_filename, permissive=True) as mrc:
        mrc_data = mrc.data.copy()  # Copying the data into memory

    print(f'\nread {mrc_filename}')
    return mrc_data

def average_frames(mrc_data, n_average):
    """
    Averages frames in blocks of n_average.
    
    Parameters:
        mrc_data (numpy.ndarray): The 3D array of shape (n_frames, height, width).
        n_average (int): Number of frames to average together.
    
    Returns:
        numpy.ndarray: The 3D array of averaged frames.
    """

    n_frames, height, width = mrc_data.shape
    print(f'\naveraging n={n_frames} SUBframes of size {width} x {height}')
    n_output_frames = n_frames // n_average
    print(f'\ninto these many FRAMES {n_output_frames}')
    averaged_frames = np.zeros((n_output_frames, height, width), dtype=np.float32)
    
    for i in range(n_output_frames):
        print(f'\nworking on output_frame {i}')
        start_idx = i * n_average
        end_idx = start_idx + n_average
        averaged_frames[i] = np.mean(mrc_data[start_idx:end_idx], axis=0)
    
    return averaged_frames

def save_to_mrc(averaged_frames, output_filename):
    """
    Saves the averaged frames to an MRC file.
    
    Parameters:
        averaged_frames (numpy.ndarray): The 3D array of averaged frames.
        output_filename (str): The filename to save the MRC file as.
    """
    with mrcfile.new(output_filename, overwrite=True) as mrc:
        mrc.set_data(averaged_frames)
        mrc.update_header_from_data()

def main():
    parser = argparse.ArgumentParser(description="Average MRC frames and save to a new MRC file.")
    parser.add_argument("--input", help="Input MRC file")
    parser.add_argument("--output", default=None, help="Output MRC file")
    parser.add_argument("--n_average", type=int, default=10, help="Number of frames to average together (default: 10)")
    
    args = parser.parse_args()

    # Read MRC file
    mrc_data = read_mrc_file(args.input)

    if args.output is None:
        args.output = args.input.replace(".mrc", "_averaged.mrc")

    # Average frames
    averaged_frames = average_frames(mrc_data, args.n_average)

    # Save to MRC file
    save_to_mrc(averaged_frames, args.output)
    print(f"Saved averaged frames to {args.output}")

if __name__ == "__main__":
    main()