"""
MIT License

Copyright (c) 2025 Ming Sun 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py # load .mat file
from mpl_toolkits.axes_grid1 import make_axes_locatable
import natsort
import os
import skimage
import scipy.io
from ipywidgets import interact, IntSlider
from skimage.exposure import match_histograms



def norm_rescale(input_):
    # normalize between 0 and 1, Min-Max Normalization
    mini = input_.min()
    maxi = input_.max()
    if mini == maxi:
        # Return a tensor of zeros if all values are the same
        print('norm case when min==max, output stays the same')
        return input_
    return (input_-mini)/(maxi-mini)

def norm_rescale_3Dslice(input_):
    """
    Min-Max Normalization for each h*w slice in a 3D tensor of shape d*h*w.
    """
    # Compute min and max for each h*w slice independently
    mini = input_.view(input_.shape[0], -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    maxi = input_.view(input_.shape[0], -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)

    # Handle case where min == max (constant slices)
    range_vals = maxi - mini
    range_vals[range_vals == 0] = 1e-8  # Avoid division by zero

    # Perform normalization
    return (input_ - mini) / range_vals

def rescale_tensor(input, target_min=-torch.pi, target_max=torch.pi):
    """
    Rescale a tensor to a specified range [target_min, target_max].
    :param input_tensor: PyTorch tensor of values to rescale
    :param target_min: Minimum value of the target range (default: -π)
    :param target_max: Maximum value of the target range (default: π)
    :return: Rescaled tensor
    """
    device, dtype = input.device, input.dtype
    # Ensure input is a PyTorch tensor
    input_tensor = torch.as_tensor(input, dtype=torch.float32).clone().detach()
    tensor_min = torch.min(input_tensor)
    tensor_max = torch.max(input_tensor)
    # Handle the case where tensor_max - tensor_min = 0
    if tensor_max - tensor_min == 0:
        # Set all values to the midpoint of the target range
        return torch.full_like(input_tensor, (target_min + target_max) / 2).to(device=device, dtype=dtype)
    # Rescale tensor to [target_min, target_max]
    rescaled_tensor = (input_tensor - tensor_min) * (target_max - target_min) / (tensor_max - tensor_min) + target_min
    return rescaled_tensor.to(device=device, dtype=dtype)


def bound(input_,bl,bu):
    #  return bounded value clipped between bl and bu
    input_ = input_*1.0 # change to float data
    input_[input_>bu] = bu
    input_[input_<bl] = bl
    return input_


def md_fftshift(x):
    device0 = x.device  # Ensure the tensor stays on the correct device
    # fftshift for multi-dimension tensor
    y = torch.zeros(x.shape, dtype=torch.complex64, device=device0)
    for dim in range(int(x.shape[0])):
        y[dim,:,:] = torch.fft.fftshift(x[dim,:,:])
    return y
    
def md_ifftshift(x):
    device0 = x.device  # Ensure the tensor stays on the correct device
    # ifftshift for multi-dimension tensor
    y = torch.zeros(x.shape, dtype=torch.complex64, device=device0)
    for dim in range(int(x.shape[0])):
        y[dim,:,:] = torch.fft.ifftshift(x[dim,:,:])
    return y

def md_fft2(x):
    device0 = x.device  # Ensure the tensor stays on the correct device
    # fftshift for multi-dimension tensor
    y = torch.zeros(x.shape, dtype=torch.complex64, device=device0)
    for dim in range(int(x.shape[0])):
        y[dim,:,:] = torch.fft.fft2(x[dim,:,:])
    return y
    
def md_ifft2(x):
    device0 = x.device  # Ensure the tensor stays on the correct device
    # ifftshift for multi-dimension tensor
    y = torch.zeros(x.shape, dtype=torch.complex64, device=device0)
    for dim in range(int(x.shape[0])):
        y[dim,:,:] = torch.fft.ifft2(x[dim,:,:])
    return y

# Define Fourier operators for multi-dimension tensor
def FT(x):
    if len(x.shape) > 2:
        return md_fftshift(md_fft2(md_ifftshift(x)))
    else: # for 2d tensor
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
def IFT(x):
    if len(x.shape) > 2:
        return md_fftshift(md_ifft2(md_ifftshift(x)))
    else: # for 2d tensor
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))

def rgb2gray(rgb):
    return torch.from_numpy(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))


def imread_8_uint(file_name):
    # plt.imread normalize the data, so we multiply 2^8-1 back
    return plt.imread(file_name)*(2**8-1)

def save_tiff_as_tensor(image_dir,name_file):
    ################ save the .tiff images as tensor file ###############
    # image_dir : Directory containing the .tif images
    # Get list of files in the directory, excluding '.' and '..'
    file_list = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith('.tiff')]
    # Sort the filenames naturally
    file_list = natsort.natsorted(file_list)
    # Initialize a list to store image data
    data = []
    # Process each image file
    for file_name in file_list:
        file_path = os.path.join(image_dir, file_name)
        image = skimage.io.imread(file_path)
        data.append(image.astype(np.double))
    # Convert list to a 3D numpy array and then to a tensor
    data_np = np.stack(data, axis=-1) #  stacks the array along the last axis. 
    data_tensor = torch.tensor(data_np, dtype=torch.double)
    # Save the tensor to a file
    output_file = f"{image_dir}\\{name_file}.pt"
    torch.save(data_tensor, output_file)
    print(f"Data saved to {output_file}")

def imread_tensor(tensor_dir,name_file):    
    ################ imread tensor file ###############
    # tensor_dir : Path to the saved .pt file
    input_file = f"{tensor_dir}\\{name_file}.pt"
    # Load the tensor from the file
    tensor_data = torch.load(input_file)
    # Verify the shape and type of the loaded tensor
    print(f"Data shape: {tensor_data.shape}")
    print(f"Data type: {tensor_data.dtype}")
    return tensor_data
        
class FP_parameter:
    def __init__(self,
                 wavelength=0.53e-6,
                pix=2.74e-6, # pixel size of the CCD
                mag=10,
                mag_image=4,
                NA=0.28,
                arraysize=14,
                LEDgap=4e-3,
                LEDheight=80e-3,
                led_index=None,
                intensity_list=None, 
                loop=4,
                dis=0.0, 
                alpha=0.005,  
                beta=0.1, 
                device=None):
        # Function to ensure input is a tensor on the correct device
        def to_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=dtype)
            else:
                return torch.tensor(x, dtype=dtype, device=self.device)
        # Select device automatically if not specified
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.wavelength = to_tensor(wavelength)
        self.pix = to_tensor(pix)
        self.mag = to_tensor(mag)
        self.mag_image = to_tensor(mag_image)
        self.NA = to_tensor(NA)
        self.arraysize = to_tensor(arraysize)
        self.LEDgap = to_tensor(LEDgap)
        self.LEDheight = to_tensor(LEDheight)
        self.dis = to_tensor(dis)
        self.alpha = to_tensor(alpha)
        self.beta = to_tensor(beta)
        self.led_index = ""
        self.loop = loop
        self.k0 = 2*torch.pi/self.wavelength
        self.spsize = self.pix/self.mag # sampling pxel size on the object plane
        self.psize = self.spsize/self.mag_image # pixel size of the HR reconstruction object or HR input image
        self.cutoff_Freq = self.NA*self.k0
    
             
def read_mat(file_path, file_name):
    """
    Load data from a .mat file, handling both HDF5 (MATLAB 7.3+) and older formats.

    Args:
        file_path (str): The directory where the .mat file is located.
        file_name (str): The name of the .mat file (without the extension).

    Returns:
        torch.Tensor: The loaded data as a PyTorch tensor.
    """

    # Construct the full file path
    full_path = os.path.join(file_path, file_name + '.mat')
    
    try:
        # Try loading as HDF5 file (MATLAB 7.3+ format)
        with h5py.File(full_path, 'r') as f:
            # Assume the dataset name matches the file name
            if file_name not in f:
                raise ValueError(f"Dataset '{file_name}' not found in the file.")
            out = np.transpose(f[file_name])  # Transpose back to its original shape
    except OSError:
        # If that fails, try loading with scipy for older .mat files
        data = scipy.io.loadmat(full_path)
        # Ignore metadata keys (e.g., '__header__', '__version__', '__globals__')
        valid_keys = [key for key in data.keys() if not key.startswith('__')]
        if not valid_keys:
            raise ValueError("No valid data keys found in the .mat file.")
        key = valid_keys[-1]  # Access the last valid key
        out = data[key]
    # Store the input dimensions
    input_dimensions = out.shape    
    # Swap the dimensions (from 400x400x196 to 196x400x400, for example)
    # Handle complex data if present
    if np.iscomplexobj(out):
        out = out.astype(np.complex64)  # Convert to complex64 for compatibility
    else:
        out = out.astype(np.float32)  # Convert to float32 for real data
    if out.ndim == 3:
        out = np.transpose(out, (2, 0, 1))  # Swap axes: (depth, height, width)
    # Store the output dimensions
    output_dimensions = out.shape
    # Convert numpy to PyTorch tensor
    out = torch.from_numpy(out)
    # Print the dimensions
    print(f"Input dimensions: {input_dimensions}")
    print(f"Output dimensions: {output_dimensions}")
    return out



def read_cell_mat(file_path, dataset_name):
    """
    Load data from a specified dataset in a .mat file. (v7.3 format)
    
    Args:
        file_path (str): The full file path of the .mat file.
        dataset_name (str): The name of the dataset to access.
        
    Returns:
        list: A list of PyTorch tensors containing the flattened data.
    """
    # Construct the full file path
    full_path = os.path.join(file_path, dataset_name + '.mat')
    
    # Load the .mat file
    with h5py.File(full_path, 'r') as f:
        # Check if the dataset exists
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in the file.")
        
        # Access the dataset containing references
        ref_dataset = f[dataset_name]

        # Initialize an empty list to store data
        data = []

        # Access each reference in the dataset
        num_cells = np.prod(ref_dataset.shape)  # Total number of cells in the array
        for idx in range(num_cells):
            # Convert flat index to multidimensional index
            multi_idx = np.unravel_index(idx, ref_dataset.shape)
            ref = ref_dataset[multi_idx]
            if isinstance(ref, h5py.h5r.Reference):
                # Dereference the reference to get the actual data
                actual_data = f[ref][:]
                data.append(torch.tensor(actual_data).transpose(0, 1))  # Read and keep the original shape

    # Convert the data to PyTorch tensors
    # tensors = [torch.tensor(arr) for arr in data]
    return data

def read_cell_mat_scipy(file_path, dataset_name):
    """
    Load data from a .mat file (non-v7.3 format) and convert it to a list of PyTorch tensors.

    Args:
        file_path (str): Path to the .mat file.
        dataset_name (str): The name of the dataset to access.

    Returns:
        list: A list of PyTorch tensors containing the data.
    """
    # Load the .mat file
    full_path = os.path.join(file_path, dataset_name + '.mat')
    mat_data = scipy.io.loadmat(full_path)

    # Check if the dataset exists
    if dataset_name not in mat_data:
        raise ValueError(f"Dataset '{dataset_name}' not found in the file.")
    
    # Access the data
    pattern = mat_data[dataset_name]

    # Convert MATLAB cell array to a Python list of PyTorch tensors
    pattern_list = []
    for cell in pattern.flatten():  # Flatten the cell array
        pattern_list.append(torch.tensor(cell))

    return pattern_list

def visualize_tensor(
    data, vmin=None, vmax=None, cmap="gray", title="Visualization"
):
    """
    Visualize a tensor or NumPy array, automatically handling GPU tensors, scaling, 
    and complex-valued data.
    
    Args:
        data (torch.Tensor or np.ndarray): Input data to visualize.
        vmin (float, optional): Minimum value for display scaling. Defaults to None (auto-scaled).
        vmax (float, optional): Maximum value for display scaling. Defaults to None (auto-scaled).
        cmap (str, optional): Colormap to use for visualization. Defaults to "gray".
        title (str, optional): Title for the plot. Defaults to "Visualization".
    """
    # Convert to NumPy if input is a tensor
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a torch.Tensor or numpy.ndarray")
    
    # Handle complex-valued data
    if np.iscomplexobj(data):
        amplitude = np.abs(data)
        phase = skimage.restoration.unwrap_phase(np.angle(data))
        
        # Plot amplitude and phase as subplots
        plt.figure(figsize=(20, 10))
        
        # Amplitude
        plt.subplot(1, 2, 1)
        plt.imshow(amplitude, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f"{title} - Amplitude")
        plt.axis("off")
        
        # Phase
        plt.subplot(1, 2, 2)
        plt.imshow(phase, cmap=cmap)
        plt.colorbar()
        plt.title(f"{title} - Phase")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        # Plot real-valued data
        plt.figure(figsize=(15, 15))
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title)
        plt.axis("off")
        plt.show()

        
        
def montage(image_set, arraysize, cmap='gray', figsize=(10, 10), wspace=0.02, hspace=0.02):
    """
    Create a montage of images in a grid layout.

    Args:
        image_set (torch.Tensor or np.ndarray): 3D tensor/array of shape (num_images, height, width).
        arraysize (int): Size of the montage grid (arraysize x arraysize).
        cmap (str): Colormap for displaying the images (default: 'gray').
        figsize (tuple): Figure size for the montage (default: (10, 10)).
        wspace (float): Space between subplots along the width (default: 0.02).
        hspace (float): Space between subplots along the height (default: 0.02).
    """
    # Convert PyTorch tensor to NumPy array, if necessary
    if isinstance(image_set, torch.Tensor):
        if image_set.device != torch.device('cpu'):
            image_set = image_set.cpu()  # Move to CPU if on GPU
        image_set = image_set.numpy()  # Convert to NumPy
    
    # Ensure the input is a NumPy array
    if not isinstance(image_set, np.ndarray):
        raise ValueError("Input image_set must be a PyTorch tensor or a NumPy array.")

    # Validate the number of images matches the grid size
    if image_set.shape[0] != arraysize**2:
        raise ValueError(f"Expected {arraysize**2} images for a {arraysize}x{arraysize} grid, "
                         f"but got {image_set.shape[0]} images.")
    
    # Create the subplot grid
    fig, axs = plt.subplots(arraysize, arraysize, figsize=figsize)
    
    # Plot each image in the grid
    for ax, i in zip(axs.flat, range(image_set.shape[0])):
        ax.imshow(image_set[i, :, :], cmap=cmap)
        ax.axis('off')  # Remove axis ticks and labels

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()

def interactive_video_viewer(video, vmin=0, vmax=255, cmap='gray'):
    """
    Display an interactive slider to view frames of a video (3D tensor or array).
    
    Args:
        video (torch.Tensor or np.ndarray): The video data, shape (num_frames, height, width).
        vmin (float): Minimum value for imshow (contrast adjustment).
        vmax (float): Maximum value for imshow (contrast adjustment).
        cmap (str): Colormap for imshow (default: 'gray').
    """
    # Ensure the video is a NumPy array
    if isinstance(video, torch.Tensor):
        video = video.numpy()

    # Function to display a specific frame
    def view_frame(frame):
        plt.figure(figsize=(6, 6))
        plt.imshow(video[frame], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f"Frame: {frame}")
        plt.axis('off')
        plt.show()

    # Create an interactive slider
    interact(view_frame, frame=IntSlider(min=0, max=video.shape[0]-1, step=1, value=0))

def interactive_complex_video_viewer(video, vmin_amp=0, vmax_amp=255, vmin_phase=-np.pi, vmax_phase=np.pi, cmap_amp='gray', cmap_phase='twilight'):
    """
    Display an interactive slider to view frames of a complex video (3D complex tensor or array).
    
    Args:
        video (torch.Tensor or np.ndarray): The complex video data, shape (num_frames, height, width), dtype=complex.
        vmin_amp (float): Minimum value for amplitude display (contrast adjustment).
        vmax_amp (float): Maximum value for amplitude display (contrast adjustment).
        vmin_phase (float): Minimum value for phase display (contrast adjustment).
        vmax_phase (float): Maximum value for phase display (contrast adjustment).
        cmap_amp (str): Colormap for amplitude (default: 'gray').
        cmap_phase (str): Colormap for phase (default: 'twilight').
    """
    # Ensure the video is a NumPy array
    if isinstance(video, torch.Tensor):
        video = video.numpy()

    # Function to display a specific frame
    def view_frame(frame):
        amplitude = np.abs(video[frame])
        phase = skimage.restoration.unwrap_phase(np.angle(video[frame]))

        plt.figure(figsize=(12, 6))

        # Plot amplitude
        plt.subplot(1, 2, 1)
        plt.imshow(amplitude, cmap=cmap_amp, vmin=vmin_amp, vmax=vmax_amp)
        plt.title(f"Amplitude - Frame: {frame}")
        plt.axis('off')

        # Plot phase
        plt.subplot(1, 2, 2)
        plt.imshow(phase, cmap=cmap_phase, vmin=vmin_phase, vmax=vmax_phase)
        plt.title(f"Phase - Frame: {frame}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Create an interactive slider
    interact(view_frame, frame=IntSlider(min=0, max=video.shape[0]-1, step=1, value=0))

def pad_with_flip(img, padding_size):
    """
    Pads the input tensor manually using flipping instead of 'replicate' mode,
    ensuring a consistent flipping pattern across all dimensions.

    Args:
        img (torch.Tensor): Input 2D or 3D complex tensor.
        padding_size (int): Number of pixels to pad on each side.

    Returns:
        torch.Tensor: Padded tensor with flipped edges.
    """
    if img.ndim == 2:  # 2D case (Height, Width)
        # Left and right flips
        left = torch.flip(img[:, 1:padding_size+1], dims=[1])  # Flip first padding_size pixels (excluding edge)
        right = torch.flip(img[:, -padding_size-1:-1], dims=[1])  # Flip last padding_size pixels (excluding edge)
        padded_img = torch.cat([left, img, right], dim=1)  # Pad horizontally

        # Top and bottom flips
        top = torch.flip(padded_img[1:padding_size+1, :], dims=[0])  # Flip first padding_size rows
        bottom = torch.flip(padded_img[-padding_size-1:-1, :], dims=[0])  # Flip last padding_size rows
        padded_img = torch.cat([top, padded_img, bottom], dim=0)  # Pad vertically

    elif img.ndim == 3:  # 3D case (Depth, Height, Width)
        # Left and right flips
        left = torch.flip(img[:, :, 1:padding_size+1], dims=[2])  # Flip left padding_size pixels (excluding edge)
        right = torch.flip(img[:, :, -padding_size-1:-1], dims=[2])  # Flip right padding_size pixels (excluding edge)
        padded_img = torch.cat([left, img, right], dim=2)  # Pad horizontally

        # Top and bottom flips
        top = torch.flip(padded_img[:, 1:padding_size+1, :], dims=[1])  # Flip top padding_size rows
        bottom = torch.flip(padded_img[:, -padding_size-1:-1, :], dims=[1])  # Flip bottom padding_size rows
        padded_img = torch.cat([top, padded_img, bottom], dim=1)  # Pad vertically

        # Front and back flips (for depth)
        front = torch.flip(padded_img[1:padding_size+1, :, :], dims=[0])  # Flip front padding_size slices
        back = torch.flip(padded_img[-padding_size-1:-1, :, :], dims=[0])  # Flip back padding_size slices
        padded_img = torch.cat([front, padded_img, back], dim=0)  # Pad along depth

    else:
        raise ValueError("Only 2D and 3D tensors are supported.")

    return padded_img

def histogram_matching(input_tensor, index_frame):
    """
    Perform histogram matching for a 3D tensor (d x w x h ) using a reference frame.
    
    Args:
        input_tensor (torch.Tensor): Input video tensor with shape (d, w, h).
        index_frame (int): Index of the reference frame along the depth dimension.
    
    Returns:
        torch.Tensor: Output tensor after histogram matching with shape (d, w, h).
    """
    # Ensure the input is a NumPy array
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.numpy()

    # Extract the reference histogram from the specified frame
    reference_hist = input_tensor[index_frame, :, :]
    min_ref = reference_hist.min()
    max_ref = reference_hist.max()

    # Initialize the output tensor
    output = np.zeros_like(input_tensor)

    # Iterate through each frame in the depth dimension
    for i in range(input_tensor.shape[0]):
        segment = bound(input_tensor[i, :, :], min_ref, max_ref)
        # Rescale, match histograms, and then rescale back to the original range
        matched = match_histograms(
            norm_rescale(segment),
            norm_rescale(reference_hist)
        )
        output[i, :, :] = rescale_tensor(matched, min_ref, max_ref)

    # Convert the output back to a PyTorch tensor
    return torch.tensor(output, dtype=torch.float32)


def temporal_gaussian_smooth_gpu(data, sigma_time=1.5, kernel_size=5, index_frame=2, device0=None):
    """
    Perform temporal Gaussian smoothing on 3D data (d x w x h) as a single segment using GPU or CPU.

    Args:
        data (torch.Tensor): Input data tensor of shape (d, w, h).
        sigma_time (float): Standard deviation of the Gaussian kernel.
        kernel_size (int): Size of the Gaussian kernel.
        index_frame (int): Reference frame index for histogram matching.
        device0 (torch.device): Device to use for computation (GPU/CPU).

    Returns:
        torch.Tensor: Smoothed data tensor of shape (d, w, h).
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device0)

    t = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device0).float()
    gaussian_kernel = torch.exp(-(t**2) / (2 * sigma_time**2))
    gaussian_kernel /= gaussian_kernel.sum()  

    gaussian_kernel = gaussian_kernel.view(1, 1, -1)

    pad_size = kernel_size // 2
    data_padded = torch.cat(
        [data[:pad_size].flip(0), data, data[-pad_size:].flip(0)], dim=0
    )  

    reshaped_data = data_padded.permute(1, 2, 0).reshape(-1, 1, data_padded.shape[0]) 
    smoothed_data = torch.nn.functional.conv1d(
        reshaped_data,  
        gaussian_kernel, 
        padding=0
    ).squeeze(1) 
    smoothed_data = smoothed_data.view(data.shape[1], data.shape[2], -1).permute(2, 0, 1)  

    smoothed_data = histogram_matching(smoothed_data.cpu(), index_frame)

    return smoothed_data

def histogram_transfer_gray_2d(input: torch.Tensor, source: torch.Tensor, eps: float = 1e-5,
):
    """
    Transfer the histogram from one grayscale image tensor to another.
    Includes normalization and rescaling, simplified for 2D grayscale input.

    Args:
        input (torch.Tensor): Input grayscale image of shape (H, W).
        source (torch.Tensor): Source grayscale image of shape (H, W).
        eps (float): Small value for numerical stability.
            Default: 1e-5.

    Returns:
        torch.Tensor: Grayscale input image matched to the source histogram.
    """
    assert input.dim() == 2, "Input must be 2D."
    assert source.dim() == 2, "Source must be 2D."

    source_min, source_max = source.min(), source.max()

    input_norm = norm_rescale(input)
    source_norm = norm_rescale(source)

    input_mean = input_norm.mean()
    input_std = input_norm.std()

    source_mean = source_norm.mean()
    source_std = source_norm.std()

    output = (input_norm - input_mean) / (input_std + eps)  
    output = output * source_std + source_mean              

    output = rescale_tensor(output, source_min, source_max) 
    
    return output  

def histogram_matching_gpu(input_tensor, index_frame, device0=None):
    """
    Perform histogram matching for a 3D tensor (d x w x h ) using a reference frame.
    
    Args:
        input_tensor (torch.Tensor): Input video tensor with shape (d, w, h).
        index_frame (int): Index of the reference frame along the depth dimension.
    
    Returns:
        torch.Tensor: Output tensor after histogram matching with shape (d, w, h).
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device0)

    reference_hist = input_tensor[index_frame, :, :]

    output = torch.zeros_like(input_tensor)

    for i in range(input_tensor.shape[0]):
        segment = input_tensor[i, :, :]
        matched = histogram_transfer_gray_2d(
            segment,
            reference_hist
        )
        output[i, :, :] = matched

    return output

def temporal_gaussian_smooth_gpu_spatial_batch(data, sigma_time=1.5, kernel_size=5, batch_size=1024, index_frame=2, device0=None):
    """
    Perform temporal Gaussian smoothing on 3D data (d x w x h) as a single segment using GPU with spatial batch processing.

    Args:
        data (torch.Tensor): Input data tensor of shape (d, w, h).
        sigma_time (float): Standard deviation of the Gaussian kernel.
        kernel_size (int): Size of the Gaussian kernel.
        batch_size (int): Number of spatial pixels (w*h) to process per batch.
        index_frame (int): Reference frame index for histogram matching.
        device0 (torch.device): Device to use for computation (GPU/CPU).

    Returns:
        torch.Tensor: Smoothed data tensor of shape (d, w, h).
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device0)

    t = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device0).float()
    gaussian_kernel = torch.exp(-(t**2) / (2 * sigma_time**2))
    gaussian_kernel /= gaussian_kernel.sum()  

    gaussian_kernel = gaussian_kernel.view(1, 1, -1)  

    pad_size = kernel_size // 2
    data_padded = torch.cat(
        [data[:pad_size].flip(0), data, data[-pad_size:].flip(0)], dim=0
    )  

    reshaped_data = data_padded.permute(1, 2, 0).reshape(-1, 1, data_padded.shape[0])  
    
    smoothed_data = torch.empty((reshaped_data.shape[0], 1, data.shape[0]), device=device0)  
    for i in range(0, reshaped_data.shape[0], batch_size):
        batch = reshaped_data[i:i+batch_size]  
        smoothed_batch = torch.nn.functional.conv1d(
            batch, 
            gaussian_kernel,  
            padding=0
        )
        smoothed_data[i:i+batch_size, :, :] = smoothed_batch 

    smoothed_data = smoothed_data.squeeze(1).reshape(data_padded.shape[1], data_padded.shape[2], -1).permute(2, 0, 1) 

    smoothed_data = histogram_matching_gpu(smoothed_data, index_frame)

    return smoothed_data


def LED_coordinate(FP, device0=None):
    
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    arraysize = int(FP.arraysize)
    xlocation = torch.zeros(1, arraysize**2, device=device0)
    ylocation = torch.zeros(1, arraysize**2, device=device0)
    # Check if arraysize is even
    if arraysize % 2 == 0:
        for i in range(arraysize):
            start_index = arraysize * i
            end_index = arraysize * (i + 1)
        
            x_values = torch.arange(-arraysize / 2 + 0.5, arraysize / 2, step=1, device=device0) * FP.LEDgap
            xlocation[0, start_index:end_index] = x_values
        
            y_value = (arraysize / 2 - 0.5 - i) * FP.LEDgap
            ylocation[0, start_index:end_index] = torch.full((arraysize,), y_value, device=device0)
    else:
        for i in range(arraysize):
            start_index = arraysize * i
            end_index = arraysize * (i + 1)
        
            x_values = torch.arange(-(arraysize - 1) / 2, (arraysize - 1) / 2 + 1, step=1, device=device0) * FP.LEDgap
            xlocation[0, start_index:end_index] = x_values
        
            y_value = ((arraysize - 1) / 2 - i) * FP.LEDgap
            ylocation[0, start_index:end_index] = torch.full((arraysize,), y_value, device=device0)
    return xlocation.flatten(), ylocation.flatten()


def FP_forward(FP,HR_object, device0=None):
        
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    xlocation,ylocation = LED_coordinate(FP)  
    dd = torch.sqrt(xlocation**2 + ylocation**2 + FP.LEDheight**2)
    kx_relative = xlocation / dd
    ky_relative = ylocation / dd
    kx = FP.k0 * kx_relative
    ky = FP.k0 * ky_relative

    m, n = HR_object.shape 
    m1 = int(torch.round(m / (FP.spsize / FP.psize)).item())  
    n1 = int(torch.round(n / (FP.spsize / FP.psize)).item())

    kmax = torch.pi / FP.spsize 
    kxm = torch.linspace(-kmax, kmax, n1, device=device0)
    kym = torch.linspace(-kmax, kmax, m1, device=device0)
    kxm, kym = torch.meshgrid(kxm, kym, indexing='xy')
    CTF = (kxm**2 + kym**2) < (FP.cutoff_Freq**2) 
    CTF = CTF.to(torch.float32).to(device0)  

    dkx = 2 * torch.pi / (FP.psize * n)
    dky = 2 * torch.pi / (FP.psize * m)
    
    # forward model 
    objectFT = FT(HR_object.to(device0))
    imSeqLowRes = torch.zeros((int(FP.arraysize)**2), m1, n1, dtype=torch.float, device=device0)
    for i in range(int(FP.arraysize)**2):  
        kxc = torch.round((n+1)/2 + kx[i].item()/dkx)
        kyc = torch.round((m+1)/2 + ky[i].item()/dky)
        ky1 = torch.round(kyc - (m1-1)/2 - 1).long() 
        kyh = ky1 + m1 - 1
        kx1 = torch.round(kxc - (n1-1)/2 - 1).long()
        kxh = kx1 + n1 - 1
        imSeqLowFT = (m1/m)**2 * objectFT[ky1:kyh+1, kx1:kxh+1] * CTF
        imSeqLowRes[i, :, :] = torch.abs(IFT(imSeqLowFT))
    LR_data = imSeqLowRes.pow(2) 
    return LR_data

 
def led_order(FP, device0=None):
    
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    xlocation,ylocation = LED_coordinate(FP)  
    max_leds_per_ring = 50  
    led_index_by_ring = torch.full((int(FP.arraysize), max_leds_per_ring), -1, dtype=torch.long, device=device0) 
    led_radius = torch.sqrt(xlocation**2 + ylocation**2).flatten()
    for ring_number in range(int(FP.arraysize)):
        if int(FP.arraysize) % 2 == 0:
            ring_radius_min = (ring_number) * FP.LEDgap
            ring_radius_max = (ring_number+1) * FP.LEDgap
        else:
            ring_radius_min = ring_number * FP.LEDgap - FP.LEDgap / 2
            ring_radius_max = ring_number * FP.LEDgap + FP.LEDgap / 2
        in_ring_indices = torch.where((led_radius >= ring_radius_min) & (led_radius < ring_radius_max))[0]
        angles = torch.atan2(ylocation[in_ring_indices], xlocation[in_ring_indices])
        sorted_indices = in_ring_indices[torch.argsort(angles)]
        num_leds = sorted_indices.shape[0]
        led_index_by_ring[ring_number, :num_leds] = sorted_indices
    led_index = led_index_by_ring[led_index_by_ring != -1]
    return led_index


def tv_regularization(image, weight=0.1):
    """
    Total Variation (TV) regularization to enforce spatial smoothness.
    Args:
        image (torch.Tensor): Input tensor (2D or batch of 2D tensors).
        weight (float): Weight of the regularization.
    Returns:
        torch.Tensor: Regularized image.
    """
    diff_x = torch.diff(image, dim=1, append=image[:, -1:])
    diff_y = torch.diff(image, dim=0, append=image[-1:, :])

    regularized_image = image - weight * (torch.sign(diff_x) + torch.sign(diff_y))
    return regularized_image

def apply_tv_regularization(image, weight=1e-5):
    amplitude = torch.abs(image)
    phase = torch.angle(image)
    amplitude_tv = tv_regularization(amplitude,weight)
    phase_tv = tv_regularization(phase,weight)
    return amplitude_tv * torch.exp(1j * phase_tv)

def FP_forward_abberation_model(FP,HR_object,pupil_function=None,device0=None):
        
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    xlocation,ylocation = LED_coordinate(FP)  
    dd = torch.sqrt(xlocation**2 + ylocation**2 + FP.LEDheight**2)
    kx_relative = xlocation / dd
    ky_relative = ylocation / dd
    kx = FP.k0 * kx_relative
    ky = FP.k0 * ky_relative

    m, n = HR_object.shape 
    m1 = int(torch.round(m / (FP.spsize / FP.psize)).item())  
    n1 = int(torch.round(n / (FP.spsize / FP.psize)).item())

    kmax = torch.pi / FP.spsize 
    kxm = torch.linspace(-kmax, kmax, n1, device=device0)
    kym = torch.linspace(-kmax, kmax, m1, device=device0)
    kxm, kym = torch.meshgrid(kxm, kym, indexing='xy')
    
    CTF = (kxm**2 + kym**2) < (FP.cutoff_Freq**2) 
    CTF = CTF.to(torch.float32).to(device0)  
    if pupil_function is None:
        CTF = (kxm**2 + kym**2) < (FP.cutoff_Freq**2)  
        pupil_function = CTF.to(torch.complex64).to(device0)
    else:
        pupil_function = pupil_function.to(device0)  

    dkx = 2 * torch.pi / (FP.psize * n)
    dky = 2 * torch.pi / (FP.psize * m)

    objectFT = FT(HR_object.to(device0))
    imSeqLowRes = torch.zeros((int(FP.arraysize)**2), m1, n1, dtype=torch.float, device=device0)
    for i in range(int(FP.arraysize)**2):  
        kxc = torch.round((n+1)/2 + kx[i].item()/dkx)
        kyc = torch.round((m+1)/2 + ky[i].item()/dky)
        ky1 = torch.round(kyc - (m1-1)/2 - 1).long() 
        kyh = ky1 + m1 - 1
        kx1 = torch.round(kxc - (n1-1)/2 - 1).long()
        kxh = kx1 + n1 - 1
        imSeqLowFT = (m1/m)**2 * objectFT[ky1:kyh+1, kx1:kxh+1] * pupil_function
        imSeqLowRes[i, :, :] = torch.abs(IFT(imSeqLowFT))
    LR_data = imSeqLowRes.pow(2) 
    return LR_data


def FP_multiplexed(FP, input, am_ph, pattern_mat, device0=None): 
    
    """
    Placeholder for the multiplexed Fourier Ptychography solver.
    Users should replace this with their own implementation (e.g., multiplexed FP,
    single-shot, or other algorithms).
    ref: https://github.com/Waller-Lab/FPM ; 
    https://github.com/Waller-Lab/DPC_withAberrationCorrection ; 
    https://github.com/Biomedical-Imaging-Group/perturbative-fpm; etc
    """
    print(
        "[Warning] FP_multiplexed called: this is a placeholder. "
        "Please implement your own reconstruction function."
        )

    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
    am_ph = am_ph.to(device0)  
    
    # Check if input is 2D or 3D
    if input.ndim == 2: # 2D tensor
        input = input.unsqueeze(0)
        q1, m1, n1 = input.shape # Rows and columns
    elif input.ndim == 3: # 3D tensor
        q1, m1, n1 = input.shape # Depth, rows, and columns 
    else:
        raise ValueError("Input must be a 2D or 3D tensor")
    m = int(torch.round(m1 * (FP.spsize / FP.psize)).item()) # rows and columns 
    n = int(torch.round(n1 * (FP.spsize / FP.psize)).item())
    
    objectRecover_abs = torch.nn.functional.interpolate(torch.abs(am_ph).unsqueeze(0).unsqueeze(0), # Add batch and channel dimensions
                        size=(m, n), mode='bilinear', align_corners=True).squeeze() 
    objectRecover_phase = torch.nn.functional.interpolate(torch.angle(am_ph).unsqueeze(0).unsqueeze(0), # Add batch and channel dimensions
                        size=(m, n), mode='bilinear', align_corners=True).squeeze() 
    return objectRecover_abs * torch.exp(1j*objectRecover_phase)




def imscale(input,color,correction_mat):
    [m1,n1,p1] = input.size()
    scale_factor = torch.tensor(1/2)
    output_dsamp = torch.zeros(int(m1*scale_factor),int(n1*scale_factor),p1) # down sample
    input_color_corrected = torch.zeros(m1,n1,p1)
    for p in range(p1):
        for i in range(int(m1*scale_factor)):
            for j in range(int(n1*scale_factor)):
                rgb_vec = torch.tensor([[input[2*i,2*j,p],(input[2*i,2*j+1,p]+input[2*i+1,2*j,p])/2,input[2*i+1,2*j+1,p]]], dtype=torch.float64)
                input_color_corrected[2*i,2*j,p] = torch.sum(correction_mat[0,:]*rgb_vec)
                input_color_corrected[2*i+1,2*j,p] = torch.sum(correction_mat[1,:]*rgb_vec)
                input_color_corrected[2*i,2*j+1,p] = torch.sum(correction_mat[1,:]*rgb_vec)
                input_color_corrected[2*i+1,2*j+1,p] = torch.sum(correction_mat[2,:]*rgb_vec)
                if color == "r":
                    output_dsamp[i,j,p] = input_color_corrected[2*i,2*j,p]
                elif color == "g":
                    output_dsamp[i,j,p] = input_color_corrected[2*i,2*j+1,p]+input_color_corrected[2*i+1,2*j,p]
                else:
                    output_dsamp[i,j,p] = input_color_corrected[2*i+1,2*j+1,p]
    return output_dsamp


def pattern_show(pattern_mat):
    # Stack the list of tensors into a single tensor
    pattern_mat_tensor = torch.stack(pattern_mat)
    # pattern_mat: each row is a pattern
    p1, p2 = pattern_mat_tensor.shape
    FP0 = FP_parameter()
    FP0.arraysize = int(np.sqrt(p2))
    FP0.LEDgap = 4e-3
    xlocation, ylocation = LED_coordinate(FP0, device0="cpu")
    # Determine the number of columns (4) and rows required
    num_cols = 4
    num_rows = int(np.ceil(p1 / num_cols))
    # Create subplots with the appropriate number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axs = axs.flatten()  # Flatten to make indexing easier
    for i in range(p1):
        ax = axs[i] 
        dot_size = 60
        bk_fig = ax.scatter(xlocation, ylocation, s=dot_size, color='black')
        ax.scatter(xlocation[pattern_mat_tensor[i, :] == 1], 
                   ylocation[pattern_mat_tensor[i, :] == 1], s=dot_size, color=[0.4660, 0.7740, 0.2880])
        ax.set_xlim([xlocation[0] * 1.1, -xlocation[0] * 1.1])
        ax.set_ylim([-ylocation[0] * 1.1, ylocation[0] * 1.1])
        ax.set_aspect('equal', 'box')
        ax.set_title(f'Pattern {i+1}')
        ax.axis('off')  # Turn off the axis

    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust space between subplots
    plt.show()

# Function to add an image and colorbar to a subplot
def add_image(ax, data, title, cmap):
    im = ax.imshow(data, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(title)
    ax.axis('off')


def mask_gen(point_num, device0=None):
    # generate a 0&1 mask
    # gaussian distribution
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if point_num <= 1:
        mask = torch.tensor(1.0, device=device0)
    else:
        sigma = 10
        rou = 0
        x_max = 10
        x, y = torch.meshgrid(
            torch.linspace(-x_max, x_max, point_num, device=device0),
            torch.linspace(-x_max, x_max, point_num, device=device0),
            indexing='ij'
        )
        coef = 1 / (2 * np.pi * sigma**2 * np.sqrt(1 - rou**2))
        exp_coef = -1 / (2 * sigma**2 * (1 - rou**2))
        Gaussian_filter = coef * torch.exp(exp_coef * (x**2 - 2 * rou * x * y + y**2))
        mask = norm_rescale(Gaussian_filter).to(device0)
    return mask

def dpc_trans_func(P, S):
    """
    Computes the phase transfer function of DPC given the source distribution.
    
    Parameters:
    P: torch.Tensor - pupil function (complex tensor)
    S: torch.Tensor - source function (complex tensor)
    
    Returns:
    Hr: torch.Tensor - transfer function of real part
    Hi: torch.Tensor - transfer function of imaginary part
    """
    
    
    FSPc = torch.conj(FT(S * P))
    FP_ = FT(P)
    
    Hr = 2 * IFT(torch.real(FSPc * FP_))
    Hi = -2 * IFT(torch.imag(FSPc * FP_))
    
    Htot = torch.sqrt(torch.abs(Hr)**2 + torch.abs(Hi)**2)
    Htotmax = Htot.max()
    Hr = Hr / Htotmax
    Hi = Hi / Htotmax
    
    return Hr, Hi

def DPC_tik_multi_pat(Idpc, H, reg, comp_fac=2.0):
    """
    DPC_TIK recover phase ph based on DPC data Idpc, with transfer function H
    and regularization parameter reg
    ph = sum_i (H_i^* F(Idpc_i))/(sum_i(abs(H_i)^2)+reg

    Parameters:
    Idpc: torch.Tensor - 3D tensor representing the DPC data
    H: torch.Tensor - 3D tensor representing the transfer function
    reg: float - regularization parameter

    Returns:
    ph: torch.Tensor - recovered phase
    """
    if Idpc.shape == H.shape:
        numerator = torch.sum(FT(Idpc) * torch.conj(H), dim=0)
        denominator = torch.sum(torch.abs(H) ** 2, dim=0) + reg
        ph = torch.real(IFT(numerator / denominator))*comp_fac
    else:
        raise ValueError('DPC data should have the same dimension as the transfer function')
    return ph

def illu_gen(pattern_mat, FP, dim, mask_half_len, device0=None):
    
    # Set default device
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LEDheight = FP.LEDheight
    pix = FP.pix  
    mag = FP.mag 
    wavelength = FP.wavelength
    spsize = pix / mag  
    xlocation,ylocation = LED_coordinate(FP) 
    xlocation, ylocation = xlocation.to(device0), ylocation.to(device0)
    dd = torch.sqrt(xlocation**2 + ylocation**2 + LEDheight**2).to(device0)
    cos_thetax = xlocation / dd
    cos_thetay = ylocation / dd
    led_fx = -cos_thetax / wavelength
    led_fy = -cos_thetay / wavelength

    umax = 1 / (2 * spsize)
    dv = 1 / (dim[0] * spsize)
    du = 1 / (dim[1] * spsize)
    fx = torch.linspace(-umax, umax - du, dim[1], device=device0)
    fy = torch.linspace(-umax, umax - dv, dim[0], device=device0)
    

    mask = mask_gen(2 * mask_half_len + 1, device0)
    S_illu = torch.zeros(dim, device=device0)
    led_used = torch.where(pattern_mat[0] == 1)[0]
    for j in led_used:
        freq_x = led_fx[j]
        freq_y = led_fy[j]
        order1 = torch.argsort(torch.abs(fx - freq_x))
        order2 = torch.argsort(torch.abs(fy - freq_y))
        range1 = torch.arange(order2[0] - mask_half_len, order2[0] + mask_half_len + 1)
        range2 = torch.arange(order1[0] - mask_half_len, order1[0] + mask_half_len + 1)
        range1 = bound(range1, mask_half_len, dim[0] - mask_half_len - 1)
        range2 = bound(range2, mask_half_len, dim[1] - mask_half_len - 1)
        S_illu[range1.long()[:, None], range2.long()[None, :]] += mask
    led_used = torch.where(pattern_mat[1] == 1)[0]
    for j in led_used:
        freq_x = led_fx[j]
        freq_y = led_fy[j]
        order1 = torch.argsort(torch.abs(fx - freq_x))
        order2 = torch.argsort(torch.abs(fy - freq_y))
        range1 = torch.arange(order2[0] - mask_half_len, order2[0] + mask_half_len + 1)
        range2 = torch.arange(order1[0] - mask_half_len, order1[0] + mask_half_len + 1)
        range1 = bound(range1, mask_half_len, dim[0] - mask_half_len - 1)
        range2 = bound(range2, mask_half_len, dim[1] - mask_half_len - 1)
        S_illu[range1.long()[:, None], range2.long()[None, :]] -= mask
    S_illu[S_illu > 0] = 1

    S_illu[S_illu < 0] = -1
    
    return S_illu

def phase_gen_2pat(dpc_data,pattern_mat_4,FP,mask_half_len,reg2):

    pix = FP.pix 
    mag = FP.mag 
    spsize = pix / mag 

    m1, n1, _ = dpc_data.shape
    kmax = 2 * np.pi / spsize / 2
    kxm = torch.linspace(-kmax, kmax, n1)
    kym = torch.linspace(-kmax, kmax, m1)
    kxm, kym = torch.meshgrid(kxm, kym, indexing='xy')
    CTF = (kxm**2 + kym**2) < (FP.cutoff_Freq**2) 
    CTF = CTF.to(torch.float32) 
    phC = torch.ones((m1, n1), dtype=torch.complex64)
    aberration = torch.ones((m1, n1), dtype=torch.complex64)
    pupil = CTF * phC * aberration
    
    pattern_mat_up_botm = pattern_mat_4[0:2]
    pattern_mat_lf_rht = pattern_mat_4[2:4]
    S1 = illu_gen(pattern_mat_up_botm, FP, (m1, n1), mask_half_len)
    S2 = illu_gen(pattern_mat_lf_rht, FP, (m1, n1), mask_half_len)
    _, Hi1 = dpc_trans_func(pupil, S1)
    _, Hi2 = dpc_trans_func(pupil, S2)
    Hi = torch.stack((Hi1, Hi2), dim=-1)
    IDPC = torch.zeros((m1, n1, 2), dtype=torch.complex64)
    IDPC[:, :, 0] = (dpc_data[:, :, 0] - dpc_data[:, :, 1]) / (dpc_data[:, :, 0] + dpc_data[:, :, 1])
    IDPC[:, :, 1] = -(dpc_data[:, :, 2] - dpc_data[:, :, 3]) / (dpc_data[:, :, 2] + dpc_data[:, :, 3])
    ph_dpc = DPC_tik_multi_pat(IDPC, Hi, reg2)

    return ph_dpc

def phase_gen_multi_pat(dpc_data,pattern_mat,FP,mask_half_len,reg2, device0=None):
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpc_data = dpc_data.to(device0)
    if isinstance(pattern_mat, list):
        pattern_mat = torch.stack(pattern_mat).to(device0)
    else:
        pattern_mat = pattern_mat.to(device0)
        
    pix = FP.pix 
    mag = FP.mag 
    spsize = pix / mag 

    p1, m1, n1 = dpc_data.shape
    kmax = 2 * np.pi / spsize / 2
    kxm = torch.linspace(-kmax, kmax, n1, device=device0)
    kym = torch.linspace(-kmax, kmax, m1, device=device0)
    kxm, kym = torch.meshgrid(kxm, kym, indexing='xy')
    CTF = (kxm**2 + kym**2) < (FP.cutoff_Freq**2) 
    CTF = CTF.to(torch.float32).to(device0) 
    phC = torch.ones((m1, n1), dtype=torch.complex64, device=device0)
    aberration = torch.ones((m1, n1), dtype=torch.complex64, device=device0)
    pupil = CTF * phC * aberration
    
    S_ = torch.zeros((int(p1/2)), m1, n1, dtype=torch.complex64, device=device0)
    Hi = torch.zeros((int(p1/2)), m1, n1, dtype=torch.complex64, device=device0)
    for i in range(int(p1/2)):
        S_[i,:,:] = illu_gen(pattern_mat[2*i:2*i+2], FP, (m1, n1), mask_half_len).to(device0)
        _, Hi[i,:,:] = dpc_trans_func(pupil, S_[i,:,:])
    
    IDPC = torch.zeros((int(p1/2), m1, n1), dtype=torch.complex64, device=device0)
    for i in range(int(p1/2)):
        if i % 2 != 0:
            IDPC[i, :, :] = (dpc_data[2*i, :, :] - dpc_data[2*i+1, :, :]) / (dpc_data[2*i, :, :] + dpc_data[2*i+1, :, :])
        else:
            IDPC[i, :, :] = -(dpc_data[2*i, :, :] - dpc_data[2*i+1, :, :]) / (dpc_data[2*i, :, :] + dpc_data[2*i+1, :, :])
            
    ph_dpc = DPC_tik_multi_pat(IDPC, Hi, torch.tensor(reg2).to(device0))

    return ph_dpc
    
def pattern_dpc(NA_illu, FP, theta):
    LEDheight = FP.LEDheight
    wavelength = FP.wavelength
    
    xlocation,ylocation = LED_coordinate(FP)
    dd = torch.sqrt(xlocation**2 + ylocation**2 + LEDheight**2)
    cos_thetax = xlocation / dd
    cos_thetay = ylocation / dd
    led_fx = -cos_thetax / wavelength
    led_fy = -cos_thetay / wavelength
    
    S0 = torch.sqrt(led_fx**2+led_fy**2)*wavelength <= NA_illu
    theta_rad = torch.deg2rad(torch.tensor(theta, device=xlocation.device))
    x_rot = xlocation * torch.cos(theta_rad) + ylocation * torch.sin(theta_rad)
    y_rot = -xlocation * torch.sin(theta_rad) + ylocation * torch.cos(theta_rad)
    
    index1 = torch.nonzero((y_rot > 0) & S0)
    index2 = torch.nonzero((y_rot < 0) & S0)
    
    return index1, index2



def gaussian(window_size, sigma):
    """
    Create a 1D Gaussian window.
    Args:
        window_size (int): Size of the window.
        sigma (float): Standard deviation of the Gaussian distribution.
    Returns:
        torch.Tensor: 1D Gaussian window.
    """
    x = torch.arange(window_size, dtype=torch.float32)  # Create tensor of indices
    x = x - window_size // 2  # Center the indices around 0
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))  # Apply Gaussian formula
    return gauss / gauss.sum()

def create_window(window_size, channel=1, device="cpu"):
    """
    Create a 2D Gaussian window for SSIM calculation.
    Args:
        window_size (int): Size of the window.
        channel (int): Number of channels.
        device (str): Device to create the window on ("cpu" or "cuda").
    Returns:
        torch.Tensor: 2D Gaussian window tensor.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1).to(device)  # Ensure it's on the correct device
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(device)
    return window
    
    
def warp_2d(image_2d, u, v):
    """
    Perform 2D warping using displacement fields u, v for x, y axes.
    Args:
        image_2d (torch.Tensor): 2D image tensor (H, W) to be warped.
        u (torch.Tensor): Displacement in the x direction (W).
        v (torch.Tensor): Displacement in the y direction (H).
    Returns:
        torch.Tensor: Warped 3D image tensor (H, W).
    """
    H, W = image_2d.shape 
    device = image_2d.device
    u = u.to(device)
    v = v.to(device)
    image_2d = image_2d.unsqueeze(0).unsqueeze(0) 
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((X, Y), dim=-1).unsqueeze(0)  
    u_norm = 2 * (u / W-1)  
    v_norm = 2 * (v / H-1)  
    disp_grid = grid.clone()
    disp_grid[..., 0] += u_norm  
    disp_grid[..., 1] += v_norm  
    disp_grid[..., 0] = bound(disp_grid[..., 0], -1, 1)
    disp_grid[..., 1] = bound(disp_grid[..., 1], -1, 1)
    
    warped_image = torch.nn.functional.grid_sample(image_2d, disp_grid, mode='bilinear', align_corners=True)
    warped_image = warped_image.squeeze(0).squeeze(0) 
    return warped_image    
    
    
def project_params(params, bounds):
    """
    Project parameters into specified bounds.
    Args:
        params (torch.Tensor): Parameters to optimize.
        bounds (list of tuples): Bounds for each parameter [(min1, max1), (min2, max2), ...].
    """
    with torch.no_grad():
        for i, (lower, upper) in enumerate(bounds):
            params[i].clamp_(lower, upper)
            

def two_frame_histogram_matching(input_tensor, reference_hist):
    """
    Perform histogram matching for a 3D tensor (1 x w x h ) using a reference frame, d = 1
    
    Args:
        input_tensor (torch.Tensor): Input tensor with shape (1, w, h).
        reference frame: reference frame.
    
    Returns:
        torch.Tensor: Output tensor after histogram matching with shape (1, w, h).
    """
    # Ensure the input is a NumPy array
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()

    # Ensure the input is a NumPy array
    if isinstance(reference_hist, torch.Tensor):
        reference_hist = reference_hist.cpu().numpy()

    min_ref = reference_hist.min()
    max_ref = reference_hist.max()

    # Initialize the output tensor
    output = np.zeros_like(input_tensor)

    # Iterate through each frame in the depth dimension

    segment = input_tensor
    # Rescale, match histograms, and then rescale back to the original range
    matched = match_histograms(
            norm_rescale(segment),
            norm_rescale(reference_hist)
    )
    output = rescale_tensor(matched, min_ref, max_ref)

    # Convert the output back to a PyTorch tensor
    return output

def compute_divergence(p):
    """
    According to Wedel etal "An Improved Algorithm for TV-L1 optical flow"
    equations (8)-(10)
    Compute the divergence of the vector field p (2-channel tensor).
    
    Args:
        p (torch.Tensor): A tensor of shape (2, H, W), where:
            - p[0] is the x-component (horizontal gradient)
            - p[1] is the y-component (vertical gradient)
    
    Returns:
        torch.Tensor: The divergence of p, of shape (H, W).
    """
    kernel_x = torch.tensor([[-1, 1, 0]], dtype=torch.float32, device=p.device).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,1,3)
    kernel_y = torch.tensor([[-1], [1], [0]], dtype=torch.float32, device=p.device).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,3,1)

    # Apply convolution for each component
    div_x = torch.nn.functional.conv2d(p[0].unsqueeze(0).unsqueeze(0), kernel_x, padding=(0,1)).squeeze()
    div_y = torch.nn.functional.conv2d(p[1].unsqueeze(0).unsqueeze(0), kernel_y, padding=(1,0)).squeeze()

    return div_x + div_y

def structure_texture_decomposition_rof(im, theta=1/8, n_iters=50, ratio1=0.7, device=None):
    """
    Decomposes an input image into structure and texture parts using the 
    Rudin-Osher-Fatemi (ROF) method.

    Args:
        im (torch.Tensor): Input grayscale image of shape (H, W).
        theta (float): Regularization parameter.
        n_iters (int): Number of iterations.
        ratio1 (float): Ratio for blending texture and structure.
        device (str): 'cuda' or 'cpu'. If None, it will automatically detect.

    Returns:
        torch.Tensor: Blended texture output.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im = im.to(device, dtype=torch.float32)

    IS = (norm_rescale(im) - 0.5) * 2
    im = IS.clone()

    tao = 1 / 4  
    delta = tao / theta

    p = torch.zeros((2, *im.shape), dtype=torch.float32, device=device)  

    for _ in range(n_iters):
        div_p = compute_divergence(p)

        I_x = torch.nn.functional.conv2d((im + theta * div_p).unsqueeze(0).unsqueeze(0), 
                       torch.tensor([[-1, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0), padding=(0,1)).squeeze()
        I_y = torch.nn.functional.conv2d((im + theta * div_p).unsqueeze(0).unsqueeze(0), 
                       torch.tensor([[-1], [1], [0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0), padding=(1,0)).squeeze()

        p[0] += delta * I_x
        p[1] += delta * I_y

        norm_p = torch.sqrt(p[0]**2 + p[1]**2).clamp(min=1.0)  
        p[0] /= norm_p
        p[1] /= norm_p

    div_p = compute_divergence(p)

    IS = norm_rescale(im + theta * div_p)
    IT = norm_rescale(im - IS)

    blended_texture = norm_rescale(ratio1 * IS + (1 - ratio1) * IT) * 255

    return blended_texture

class joint_optimize_Params:
    def __init__(self, 
                 beta=0.1, 
                 delta1=0.5, 
                 delta2=0.1,
                 lambda0=1.1, 
                 lambda1=0.15, 
                 lambda2=10, 
                 gamma=0.001, 
                 miu2=0.05, 
                 miu3=0.001,):
        '''
        '''
        self.beta = beta
        self.delta1 = delta1
        self.delta2 = delta2
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self.miu2 = miu2
        self.miu3 = miu3
   

def warp_2d_complex(image_2d, u, v):
    """
    Perform 2D warping for complex-valued tensors using displacement fields u, v for x, y axes.
    Args:
        image_2d (torch.Tensor): 2D complex image tensor (H, W) to be warped.
        u (torch.Tensor): Displacement in the x direction (W).
        v (torch.Tensor): Displacement in the y direction (H).
    Returns:
        torch.Tensor: Warped 2D complex tensor (H, W).
    """
    # Separate real and imaginary parts
    real_part = image_2d.real
    imag_part = image_2d.imag
    
    # Warp real and imaginary parts separately
    warped_real = warp_2d(real_part, u, v)  # Use the existing warp_2d function
    warped_imag = warp_2d(imag_part, u, v)  # Use the existing warp_2d function
    
    # Recombine into a complex tensor
    warped_image = warped_real + 1j * warped_imag
    
    return warped_image

def Warp(o_t2, U, V):
    """
    Warping from the second frame to the first one using optical flow from the first frame to the second.
    Input and output are both vectors.
    
    Args:
        o_t2 (torch.Tensor): Input vector of shape (H*W,) or a complex tensor.
        U (torch.Tensor): Flow matrix for the x direction (H, W).
        V (torch.Tensor): Flow matrix for the y direction (H, W).        
    Returns:
        torch.Tensor: Warped vector.
    """
    # Reshape the input vector into a 2D tensor (H, W)
    H = int(torch.sqrt(torch.tensor(o_t2.numel())))  # Assuming the image is square
    W = H
    o_t2_mat = o_t2.view(H, W)
    # Perform warping using the main displacement fields U and V
    o_t2_warp_ = warp_2d_complex(o_t2_mat, U, V)
    # Reshape the warped matrix back into a vector
    o_t2_warp_vec = o_t2_warp_.view(-1)
    return o_t2_warp_vec


def soft_thresholding(input_tensor, threshold):
    """
    Apply soft thresholding (shrinkage) operator.

    Args:
        input_tensor (torch.Tensor): Input tensor (real or complex).
        threshold (float): Soft-thresholding value.

    Returns:
        torch.Tensor: Thresholded tensor.
    """
    return torch.sign(input_tensor) * torch.nn.functional.relu(torch.abs(input_tensor) - threshold)


def Dt_operator_gaussian(input, Time_slot, kernel_size_am, sigma_am, kernel_size_ph, sigma_ph):
    if kernel_size_am % 2 == 0 or kernel_size_ph % 2 == 0:
        raise ValueError('Kernel sizes (kernel_size_am and kernel_size_ph) must be odd.')
    kernel_am = generate_zero_sum_gaussian_kernel(kernel_size_am, sigma_am ,device0=input.device)
    kernel_ph = generate_zero_sum_gaussian_kernel(kernel_size_ph, sigma_ph ,device0=input.device)
    input_t_len = int(input.numel() // Time_slot)  
    input_am = torch.abs(input)
    input_ph = torch.angle(input)
    side_length = int(torch.sqrt(torch.tensor(input_t_len)))
    input_reshaped_am = input_am.view(Time_slot, side_length, side_length)
    input_reshaped_ph = input_ph.view(Time_slot, side_length, side_length)
    pad_size_am = kernel_size_am // 2
    pad_size_ph = kernel_size_ph // 2
    input_am_padded = torch.cat([input_reshaped_am[1:pad_size_am+1].flip(0),
                                 input_reshaped_am,
                                 input_reshaped_am[-pad_size_am-1:-1].flip(0)], dim=0)
    input_ph_padded = torch.cat([input_reshaped_ph[1:pad_size_ph+1].flip(0),
                                 input_reshaped_ph,
                                 input_reshaped_ph[-pad_size_ph-1:-1].flip(0)], dim=0)
    out_am = torch.zeros_like(input_reshaped_am)
    out_ph = torch.zeros_like(input_reshaped_ph)
    for t in range(Time_slot):
        temporal_am = input_am_padded[t:(t + kernel_size_am), :, :]
        temporal_ph = input_ph_padded[t:(t + kernel_size_ph), :, :] 
        out_am[t, :, :] = torch.sum(temporal_am * kernel_am.view(-1, 1, 1).expand_as(temporal_am), dim=0)
        out_ph[t, :, :] = torch.sum(temporal_ph * kernel_ph.view(-1, 1, 1).expand_as(temporal_ph), dim=0)
    out = out_am.flatten() * torch.exp(1j * out_ph.flatten())
    return out

def generate_zero_sum_gaussian_kernel(kernel_size, sigma, device0=None):
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tt = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device = device0)
    kernel = torch.exp(-0.5 * (tt / sigma) ** 2)
    kernel = kernel / kernel.sum()  
    central_idx = kernel_size // 2
    kernel = -kernel   
    kernel[central_idx] = -kernel.sum() + kernel[central_idx]

    return kernel

def swirl2d(img, swirl_strength=0.005, padding_size=100, device0=None):
    """
    Apply a 2D swirl effect to a complex-valued image using PyTorch grid_sample.

    Parameters:
    - img: 2D complex tensor
    - swirl_strength: Strength of the swirl effect
    - padding_size: Padding size to avoid border effects

    Returns:
    - swirled_img: 2D complex tensor after swirl transformation
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(img, torch.Tensor):
        img_torch = torch.tensor(img, dtype=torch.complex64, device=device0)
    else:
        img_torch = img.to(dtype=torch.complex64).to(device0)
    H, W = img_torch.shape
    padded_img = pad_with_flip(img_torch, padding_size)
    rows, cols = padded_img.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, rows, device=device0), torch.linspace(-1, 1, cols, device=device0), indexing='ij')
    centerX, centerY = 0, 0  
    deltaX, deltaY = x - centerX, y - centerY
    radius_map = torch.sqrt(deltaX**2 + deltaY**2)
    theta = torch.atan2(deltaY, deltaX)
    swirled_theta = theta + swirl_strength * radius_map
    new_x = radius_map * torch.cos(swirled_theta)
    new_y = radius_map * torch.sin(swirled_theta)
    new_x = torch.clamp(new_x, -1, 1).to(device0)
    new_y = torch.clamp(new_y, -1, 1).to(device0)
    grid = torch.stack((new_x, new_y), dim=-1).unsqueeze(0)  
    magnitude = torch.abs(padded_img).unsqueeze(0).unsqueeze(0)  
    phase = torch.angle(padded_img).unsqueeze(0).unsqueeze(0)  
    magnitude_swirled = torch.nn.functional.grid_sample(magnitude, grid, mode='bilinear', padding_mode='border', align_corners=True)
    phase_swirled = torch.nn.functional.grid_sample(phase, grid, mode='bilinear', padding_mode='border', align_corners=True)
    swirled_img = magnitude_swirled.squeeze().exp() * torch.exp(1j * phase_swirled.squeeze())
    crop_x1, crop_x2 = padding_size, padding_size + W
    crop_y1, crop_y2 = padding_size, padding_size + H
    swirled_img = swirled_img[crop_y1:crop_y2, crop_x1:crop_x2] 
    return swirled_img


def add_gaussian_noise(image, mean=0.0, std=0.01):
    noise = torch.randn_like(image) * std + mean
    return image + noise


class LossNormalizer:
    def __init__(self):
        """
        Initializes the loss normalizer without requiring a momentum parameter.
        Stores the initial value of each loss component to use for normalization.
        """
        self.initial_values = {}

    def normalize(self, name, loss_value):
        """
        Normalizes the given loss value by its initial value, establishing a stable scale.

        Args:
            name (str): A unique name for the loss component, used to track its initial value.
            loss_value (torch.Tensor): The current value of the loss to normalize.

        Returns:
            torch.Tensor: The normalized loss value.
        """
        if name not in self.initial_values:
            self.initial_values[name] = loss_value.detach()  # Store the initial value
        # Normalize the current loss by the initial value
        normalized_loss = loss_value / (self.initial_values[name] + 1e-8)  # Avoid division by zero

        return normalized_loss
    
    
def huber_loss(x, mu):
    """ Compute the Huber loss element-wise."""
    abs_x = torch.abs(x)
    condition = abs_x <= mu
    squared_loss = (1 / (2 * mu)) * (x ** 2)
    linear_loss = abs_x - (mu / 2)
    return torch.where(condition, squared_loss, linear_loss)

def huber_norm(x, mu):
    """ Compute the Huber norm for a tensor, which can be real or complex."""
    if torch.is_complex(x):
        return huber_norm(x.real, mu) + huber_norm(x.imag, mu)
    else:
        return torch.sum(huber_loss(x, mu))
    
def apply_huber_Dy_to_3D_tensor(input_tensor, mu, single_slice=False):
    """
    Apply the Dy operator to each 2D slice of a 3D tensor and sum the Huber norms of the results,
    processing all slices simultaneously as a batch.
    
    Args:
        input_tensor (torch.Tensor): Input 3D tensor of size (N, m, n) where each slice along the first dimension is a 2D complex image.
        mu (float): Threshold parameter for the Huber loss.
        
    Returns:
        float: Sum of Huber norms of derivatives across all slices.
    """
    dy_filter = torch.tensor([[0], [1], [-1]], dtype=torch.float32, device=input_tensor.device).unsqueeze(0).unsqueeze(0) 
    
    if single_slice:
        input_tensor = input_tensor.unsqueeze(0)  

    img_real = input_tensor.real.unsqueeze(1)  
    img_imag = input_tensor.imag.unsqueeze(1)  

    deri_real = torch.nn.functional.conv2d(
        torch.nn.functional.pad(img_real, (0, 0, 1, 1), mode='replicate'), dy_filter
    ).squeeze(1) 
    deri_imag = torch.nn.functional.conv2d(
        torch.nn.functional.pad(img_imag, (0, 0, 1, 1), mode='replicate'), dy_filter
    ).squeeze(1)  

    deri_res = deri_real + 1j * deri_imag
    
    if single_slice:
        total_huber_norm = huber_norm(deri_res.squeeze(0), mu)  
    else:
        total_huber_norm = torch.sum(torch.stack([huber_norm(slice, mu) for slice in deri_res]))

    return total_huber_norm  

def apply_huber_Dx_to_3D_tensor(input_tensor, mu, single_slice=False):
    """
    Apply the Dx operator to each 2D slice of a 3D tensor and sum the Huber norms of the results,
    processing all slices simultaneously as a batch.
    
    Args:
        input_tensor (torch.Tensor): Input 3D tensor of size (N, m, n) where each slice along the first dimension is a 2D complex image.
        mu (float): Threshold parameter for the Huber loss.
        
    Returns:
        float: Sum of Huber norms of derivatives across all slices.
    """
    # Define the filter for -y direction derivative
    dx_filter = torch.tensor([[0, -1, 1]], dtype=torch.float32, device=input_tensor.device).unsqueeze(0).unsqueeze(0) 
    
    if single_slice:
        input_tensor = input_tensor.unsqueeze(0)  
        
    img_real = input_tensor.real.unsqueeze(1)  
    img_imag = input_tensor.imag.unsqueeze(1)  

    deri_real = torch.nn.functional.conv2d(
        torch.nn.functional.pad(img_real, (1, 1, 0, 0), mode='replicate'), dx_filter
    ).squeeze(1)  
    deri_imag = torch.nn.functional.conv2d(
        torch.nn.functional.pad(img_imag, (1, 1, 0, 0), mode='replicate'), dx_filter
    ).squeeze(1)  

    deri_res = deri_real + 1j * deri_imag
    
    if single_slice:
        total_huber_norm = huber_norm(deri_res.squeeze(0), mu)  
    else:
        total_huber_norm = torch.sum(torch.stack([huber_norm(slice, mu) for slice in deri_res]))

    return total_huber_norm  

def apply_huber_Dt_to_3D_tensor(input_tensor, mu, kernel_size_am, sigma_am, kernel_size_ph, sigma_ph, device=None):
    """
    Applies a Gaussian-smoothed time derivative (Dt) operator and calculates the Huber norm of the results.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of size (N, m, n) where N is the number of time steps.
        mu (float): Threshold parameter for the Huber loss.
        kernel_size_am (int): Kernel size for smoothing amplitude, must be odd.
        sigma_am (float): Standard deviation for amplitude Gaussian kernel.
        kernel_size_ph (int): Kernel size for smoothing phase, must be odd.
        sigma_ph (float): Standard deviation for phase Gaussian kernel.
        
    Returns:
        float: Sum of Huber norms of time derivatives across all time steps.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dt = Dt_operator_gaussian_3D_tensor(input_tensor, kernel_size_am, sigma_am, kernel_size_ph, sigma_ph, device)
    total_huber_norm = huber_norm(Dt, mu) 
    return total_huber_norm  

def Dt_operator_gaussian_3D_tensor(input, kernel_size_am, sigma_am, kernel_size_ph, sigma_ph, device=None): 
    """
    Applies Gaussian smoothing to both the amplitude and phase of a 3D tensor.
    
    Args:
        input (torch.Tensor): Input tensor of size (T, h, w).
        kernel_size_am (int): Kernel size for amplitude.
        sigma_am (float): Standard deviation for amplitude kernel.
        kernel_size_ph (int): Kernel size for phase.
        sigma_ph (float): Standard deviation for phase kernel.
        device (torch.device): Device to perform computations on.
        
    Returns:
        torch.Tensor: Output tensor of smoothed complex numbers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check kernel sizes are odd
    if kernel_size_am % 2 == 0 or kernel_size_ph % 2 == 0:
        raise ValueError('Kernel sizes (kernel_size_am and kernel_size_ph) must be odd.')
    kernel_am = generate_zero_sum_gaussian_kernel(kernel_size_am, sigma_am, device)
    kernel_ph = generate_zero_sum_gaussian_kernel(kernel_size_ph, sigma_ph, device)
    input_am = torch.abs(input).to(device)
    input_ph = torch.angle(input).to(device)

    # Manual padding in the temporal dimension
    pad_size_am = kernel_size_am // 2
    pad_size_ph = kernel_size_ph // 2
    input_am_padded = torch.cat([input_am[1:pad_size_am+1].flip(0), 
                                 input_am, 
                                 input_am[-pad_size_am-1:-1].flip(0)], dim=0) 
    input_ph_padded = torch.cat([input_ph[1:pad_size_ph+1].flip(0),
                                 input_ph, 
                                 input_ph[-pad_size_ph-1:-1].flip(0)], dim=0)
    # Initialize output tensors
    out_am = torch.zeros_like(input_am)
    out_ph = torch.zeros_like(input_ph)
    for t in range(input.size(0)): 
        temporal_am = input_am_padded[t:(t + kernel_size_am), :, :] 
        temporal_ph = input_ph_padded[t:(t + kernel_size_ph), :, :]
        out_am[t, :, :] = torch.sum(temporal_am * kernel_am.view(-1, 1, 1).expand_as(temporal_am), dim=0)
        out_ph[t, :, :] = torch.sum(temporal_ph * kernel_ph.view(-1, 1, 1).expand_as(temporal_ph), dim=0)
    out = out_am * torch.exp(1j * out_ph)
    return out

def FP_multiplexed_forward(FP, Obj, device0=None): 
    """
    Sum the low-resolution (LR) data generated by specific LED patterns in a Fourier Ptychography setup.
    
    Args:
        FP: Fourier Ptychography setup object containing system parameters and LED sequence.
        Obj (torch.Tensor): High-resolution object tensor to be processed.
        device0 (torch.device, optional): Specifies the device for computation. Defaults to GPU if available, else CPU.
        
    Returns:
        torch.Tensor: Summed low-resolution data tensor.
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")               
    LR_data = FP_forward(FP,Obj.squeeze(), device0=device0) 
    out = torch.sum(LR_data[FP.led_pattern_seq, :, :], dim=0)
    # print(led_pattern_seq,led_pattern_seq_reorder)
    return out

def FP_fidelity_loss(num_time_slot, start_index, frame_num, n_p, pattern, FP, o_res_image_t, raw_data_, device0=None):
    """
    Calculate the fidelity loss for Multiplexed Fourier Ptychography based on MSE between predicted and actual images.
    
    Args:
        num_time_slot (int): Half Number of time frames.
        start_index (int): Starting index for data.
        frame_num (int): Number of frames in each time slot.
        n_p (int): Total number of patterns.
        pattern (torch.Tensor): LED pattern array used in FP imaging.
        FP (object): Fourier Ptychography setup object containing system parameters.
        o_res_image_t (torch.Tensor): High-resolution images used for reconstruction.
        raw_data_ (torch.Tensor): Actual captured data to compare against.
        device0 (torch.device, optional): Device on which to perform computations.
        
    Returns:
        float: Total fidelity loss computed across all time slots.
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    fidelity_loss = 0.0  # Initialize total loss
    for i_ref in range(num_time_slot):
        start_ref = start_index + i_ref* frame_num
        end_ref = start_ref + frame_num - 1
        pattern_th = int(torch.ceil(torch.remainder(torch.tensor(start_ref + 1, dtype=torch.float32), 
                                                torch.tensor(n_p * frame_num, dtype=torch.float32)) / frame_num)) -1 # -1 for 0-based indexing 
        pattern_mat_tmp = pattern[pattern_th]
        # Forward the multiplexed FP model and calculate the loss against the raw data
        FP.led_pattern_seq = torch.nonzero(pattern_mat_tmp[0, :] != 0).squeeze()
        predicted = FP_multiplexed_forward(FP, o_res_image_t[2*i_ref], device0=device0)
        actual = raw_data_[start_ref, :, :].to(device0)
        loss_start = torch.nn.functional.mse_loss(predicted, actual)
        
        FP.led_pattern_seq = torch.nonzero(pattern_mat_tmp[1, :] != 0).squeeze()        
        predicted = FP_multiplexed_forward(FP, o_res_image_t[2*i_ref+1], device0=device0)
        actual = raw_data_[end_ref, :, :].to(device0)
        loss_end = torch.nn.functional.mse_loss(predicted, actual)
        
        fidelity_loss += (loss_start + loss_end) 

    return fidelity_loss

def flow_loss(o_res_image_t, U_t_abs, V_t_abs):
    """
    Calculate the loss between each warped frame and the corresponding original frame
    in a sequence of complex-valued frames based on given velocity fields.
    
    Args:
        o_res_image_t (torch.Tensor): Tensor of shape (Time_slot, H, W) containing complex-valued frames.
        U_t_abs (torch.Tensor): Tensor of shape (Time_slot, H, W) containing x-direction displacements.
        V_t_abs (torch.Tensor): Tensor of shape (Time_slot, H, W) containing y-direction displacements.
    
    Returns:
        float: Total L1 loss for the sequence after warping based on velocity fields.
    """
    flow_loss = 0.0  
    Time_slot = o_res_image_t.shape[0]
    for t in range(Time_slot):
        if t == Time_slot-1:
            o_t_next = o_res_image_t[0] 
            o_t_warp = o_res_image_t[Time_slot-1] 
        else:
            o_t_next = o_res_image_t[t+1]
            o_t_warp = warp_2d_complex(o_t_next,U_t_abs[t,:,:],V_t_abs[t,:,:])
        o_t = o_res_image_t[t]
        loss_warp = torch.nn.functional.l1_loss(o_t_warp, o_t)
        flow_loss += loss_warp
    return flow_loss

def warp_projection_loss(num_time_slot, start_index, frame_num, n_p, pattern, FP, o_res_image_t, raw_data_, U_t_abs, V_t_abs, Uaff_t_abs, Vaff_t_abs, device0=None):
    """
    Calculate the fidelity loss for Multiplexed Fourier Ptychography based on MSE between predicted and actual images.
    
    Args:
        num_time_slot (int): Half Number of time frames.
        start_index (int): Starting index for data.
        frame_num (int): Number of frames in each time slot.
        n_p (int): Total number of patterns.
        pattern (torch.Tensor): LED pattern array used in FP imaging.
        FP (object): Fourier Ptychography setup object containing system parameters.
        o_res_image_t (torch.Tensor): High-resolution images used for reconstruction.
        raw_data_ (torch.Tensor): Actual captured data to compare against.
        device0 (torch.device, optional): Device on which to perform computations.
        
    Returns:
        float: Total fidelity loss computed across all time slots.
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    forward_projection_loss = 0.0  # Initialize total loss
    for i_ref in range(num_time_slot):   
        # Forward the multiplexed FP model and calculate the loss against the raw data
        start_ref = start_index + i_ref* frame_num
        end_ref = start_ref + frame_num - 1
        pattern_th = int(torch.ceil(torch.remainder(torch.tensor(start_ref + 1, dtype=torch.float32), 
                                                torch.tensor(n_p * frame_num, dtype=torch.float32)) / frame_num)) -1 
        pattern_mat_tmp = pattern[pattern_th]
        FP.led_pattern_seq = torch.nonzero(pattern_mat_tmp[1, :] != 0).squeeze()
        predicted = FP_multiplexed_forward(FP, warp_2d_complex(o_res_image_t[2*i_ref],Uaff_t_abs[2*i_ref,:,:],Vaff_t_abs[2*i_ref,:,:]),device0=device0) 
        actual = raw_data_[start_ref+1, :, :].to(device0)
        loss_start = torch.nn.functional.mse_loss(predicted, actual)
        
        if i_ref == num_time_slot-1:
            loss_end = 0.0 
        else:
            start_ref = start_index + (i_ref+1)* frame_num
            end_ref = start_ref + frame_num - 1
            pattern_th = int(torch.ceil(torch.remainder(torch.tensor(start_ref + 1, dtype=torch.float32), 
                                                    torch.tensor(n_p * frame_num, dtype=torch.float32)) / frame_num)) -1 
            pattern_mat_tmp = pattern[pattern_th]
            FP.led_pattern_seq = torch.nonzero(pattern_mat_tmp[0, :] != 0).squeeze()    
            predicted = FP_multiplexed_forward(FP, warp_2d_complex(o_res_image_t[2*i_ref+1],Uaff_t_abs[2*i_ref+1,:,:],Vaff_t_abs[2*i_ref+1,:,:]), device0=device0) 
            actual = raw_data_[start_ref, :, :].to(device0)
            loss_end = torch.nn.functional.mse_loss(predicted, actual)
        
        forward_projection_loss += (loss_start + loss_end) 
    
    backward_projection_loss = 0.0 
    for i_ref in range(num_time_slot):   
        # Backward the multiplexed FP model and calculate the loss against the raw data
        if i_ref == 0:
            loss_start = 0.0 
        else:
            start_ref = start_index + (i_ref-1) * frame_num
            end_ref = start_ref + frame_num - 1
            pattern_th = int(torch.ceil(torch.remainder(torch.tensor(start_ref + 1, dtype=torch.float32), 
                                                    torch.tensor(n_p * frame_num, dtype=torch.float32)) / frame_num)) -1 
            pattern_mat_tmp = pattern[pattern_th]
            FP.led_pattern_seq = torch.nonzero(pattern_mat_tmp[1, :] != 0).squeeze()
            predicted = FP_multiplexed_forward(FP, warp_2d_complex(o_res_image_t[2*i_ref],U_t_abs[2*i_ref-1,:,:],V_t_abs[2*i_ref-1,:,:]),device0=device0)  
            actual = raw_data_[end_ref, :, :].to(device0)
            loss_start = torch.nn.functional.mse_loss(predicted, actual)
            
        start_ref = start_index + i_ref* frame_num
        end_ref = start_ref + frame_num - 1
        pattern_th = int(torch.ceil(torch.remainder(torch.tensor(start_ref + 1, dtype=torch.float32), 
                                                torch.tensor(n_p * frame_num, dtype=torch.float32)) / frame_num)) -1 
        pattern_mat_tmp = pattern[pattern_th]
        FP.led_pattern_seq = torch.nonzero(pattern_mat_tmp[0, :] != 0).squeeze()  
        predicted = FP_multiplexed_forward(FP, warp_2d_complex(o_res_image_t[2*i_ref+1],U_t_abs[2*i_ref,:,:],V_t_abs[2*i_ref,:,:]), device0=device0)
        actual = raw_data_[start_ref, :, :].to(device0)
        loss_end = torch.nn.functional.mse_loss(predicted, actual)
        
        backward_projection_loss += (loss_start + loss_end) 

    return forward_projection_loss + backward_projection_loss


def blend_tensors(base_tensor, new_tensor, blend_weight):
    """ Blend two tensors based on a weight factor. """
    return base_tensor * (1 - blend_weight) + new_tensor * blend_weight


def flow_loss_single_slice(o_res_image_t, U_t_abs, V_t_abs):
    """
    Calculate the loss between each warped frame and the corresponding original frame
    in a sequence of complex-valued frames based on given velocity fields.
    
    Args:
        o_res_image_t (torch.Tensor): Tensor of shape (2, H, W) containing complex-valued frames.
        U_t_abs (torch.Tensor): Tensor of shape (H, W) containing x-direction displacements.
        V_t_abs (torch.Tensor): Tensor of shape (H, W) containing y-direction displacements.
    
    Returns:
        float: Total L1 loss for the sequence after warping based on velocity fields.
    """
    
    o_t_next = o_res_image_t[1]
    o_t_warp = warp_2d_complex(o_t_next,U_t_abs,V_t_abs) # warping from o_t+1 to o_t
    o_t = o_res_image_t[0]
    flow_loss = torch.nn.functional.l1_loss(o_t_warp, o_t)
    return flow_loss


def FP_fidelity_loss_single_slice(FP, o_res_image_t_slice, raw_data_slice, device0=None):
    """
    Calculate the single slice fidelity loss for Multiplexed Fourier Ptychography based on MSE between predicted and actual images.
    
    Args:
        FP (object): Fourier Ptychography setup object containing system parameters.
        o_res_image_t_slice (torch.Tensor): High-resolution image of one slice used for reconstruction.
        raw_data_slice (torch.Tensor): Actual captured data to compare against.
        device0 (torch.device, optional): Device on which to perform computations.
        
    Returns:
        float: Total fidelity loss computed across all time slots.
    """
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    predicted = FP_multiplexed_forward(FP, o_res_image_t_slice, device0=device0)
    actual = raw_data_slice.to(device0)
    fidelity_loss = torch.nn.functional.mse_loss(predicted, actual)
    return fidelity_loss