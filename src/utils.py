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


import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur

def plot_2d_flow(u,v,step=5,color='C0',scale=1, width=.005):
    # u,v 2d tensors
    # step, downsample the flow vectors
    if u.is_cuda:
        u_np = u.squeeze().detach().cpu().numpy()
        v_np = v.squeeze().detach().cpu().numpy()
    else:
        u_np = u.squeeze().detach().numpy()
        v_np = v.squeeze().detach().numpy()
    
    u_downsampled = u_np[::step, ::step]
    v_downsampled = v_np[::step, ::step]

    X, Y = np.meshgrid(np.arange(0, u_np.shape[1], step), np.arange(u_np.shape[0], 0, -step), indexing='xy')

    # plot
    fig, ax = plt.subplots()
    ax.quiver(X, Y,u_downsampled, -v_downsampled, color='C0', angles='xy',
          scale_units='xy', scale=scale, width=width)
    plt.title("Optical Flow")
    plt.show()

  

def gaussian_smoothing_2d(input_tensor: torch.Tensor, sigma: float = 0.2):
    """
    Apply Gaussian smoothing to a 2D tensor using convolution.
    Args:
    input_tensor (torch.Tensor): A 2D tensor of shape (Height, Width).
    kernel_size (int): Size of the Gaussian kernel.
    sigma (float): Standard deviation of the Gaussian filter. larger, smmother
    Returns:
    torch.Tensor: Smoothed 2D tensor.
    """
    # Ensure 4D tensor (batch size, channels, height, width)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Adding batch and channel dimensions
    # Ensure the kernel size is an odd positive number
    k_size = int(6 * sigma + 1)
    if k_size % 2 == 0:  # Ensure the kernel size is odd
        k_size += 1
    if k_size <= 0:
        raise ValueError("Kernel size must be a positive odd number.")
    gaussian_blur = GaussianBlur(kernel_size=k_size, sigma=sigma)
    smoothed_tensor = gaussian_blur(input_tensor)
    # Remove the added batch and channel dimensions
    return smoothed_tensor.squeeze()  

def median_filter_2d(input_tensor, kernel_size=3):
    """
    Apply a 2D median filter to a tensor using PyTorch.
    
    Args:
        input_tensor (torch.Tensor): A 2D tensor (H, W) or 4D tensor (B, C, H, W).
        kernel_size (int): Size of the median filter kernel, default is 3.
    
    Returns:
        torch.Tensor: The median-filtered tensor.
    """
    # Ensure input is 4D (batch, channel, height, width)
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # (1, C, H, W)
    
    # Get height and width for reshaping later
    B, C, H, W = input_tensor.shape
    
    # Pad the input to handle borders
    padding = kernel_size // 2
    input_padded = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')
    
    # Use unfold to get sliding local blocks
    unfolded = F.unfold(input_padded, kernel_size=kernel_size).view(B, C, kernel_size**2, H, W)
    
    # Compute the median along the last dimension
    median = unfolded.median(dim=2)[0]
    
    return median.squeeze()


def resize_2d_tensor(input_tensor, size=None, scale_factor=None):
    """
    Resize a 2D tensor using torch.nn.functional.interpolate.
    Args:
    input_tensor (torch.Tensor): A 2D tensor (H, W).
    size (tuple): The target output size (height, width).
    scale_factor (float or tuple): The scaling factor for each dimension.
    mode (str): Interpolation mode, 'trilinear' or 'nearest'.
    Returns:
    torch.Tensor: The resized tensor.
    """
    # Ensure 4D tensor (batch size, channels, height, width)
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    # Perform resizing
    if size is not None:
        resized_tensor = F.interpolate(input_tensor, size=size, mode='bicubic', align_corners=True)
    elif scale_factor is not None:
        resized_tensor = F.interpolate(input_tensor, scale_factor=(scale_factor, scale_factor), mode='bicubic', align_corners=True)
    else:
        raise ValueError("Either size or scale_factor must be provided.")
    # Remove batch and channel dimensions if they were added
    resized_tensor = resized_tensor.squeeze()
    return resized_tensor


def compute_2d_derivatives(input_tensor):
    """
    Compute the partial derivatives of a 2D tensor in the x, y directions.
    Args:
    input_tensor (torch.Tensor): A 2D tensor of shape (H, W) or a 4D tensor (batch, channels, H, W).
    Returns:
    Ix, Iy: Partial derivatives of the input tensor in the x, y directions.
    """
    device = input_tensor.device
    Kx = torch.tensor([1, -8, 0, 8, -1], dtype=torch.float32) / 12
    Kx = Kx.to(device).view(1, 1, 1, 5)  
    Ky = torch.tensor([1, -8, 0, 8, -1], dtype=torch.float32) / 12
    Ky = Ky.to(device).view(1, 1, 5, 1) 
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  
    Ix = F.conv2d(input_tensor, Kx, padding=(0, 2)) 
    Iy = F.conv2d(input_tensor, Ky, padding=(2, 0))
    Ix = Ix.squeeze(0).squeeze(0)
    Iy = Iy.squeeze(0).squeeze(0)
    return Ix, Iy

def bound(x, min_val, max_val):
    """
    Clamp values to be within the specified bounds.
    Args:
        x (torch.Tensor): Tensor to be clamped.
        min_val (float): Minimum value.
        max_val (float): Maximum value.  
    Returns:
        torch.Tensor: Clamped tensor.
    """
    return torch.clamp(x, min=min_val, max=max_val).to(x.device) 


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
    
    warped_image = F.grid_sample(image_2d, disp_grid, mode='bilinear', align_corners=True)
    warped_image = warped_image.squeeze(0).squeeze(0) 
    return warped_image


def polar_mag_ft(Img):
    """
    Apply a polar coordinates transfer of 2d image.
    Args:
        Img (torch.Tensor): A 2D tensor of shape (H, W).
    Returns:
        torch.Tensor: The 2D tensor of new coordinates.
    """
    device = Img.device
    row, col = Img.shape
    cen_x = (col + 1) / 2
    cen_y = (row + 1) / 2
    theta = torch.linspace(0, 2 * torch.pi, row+1, device=device)
    theta = theta[0:row]
    d = torch.tensor(min(col - cen_x, row - cen_y), dtype=torch.float32, device=device)
    d = torch.clamp(d, min=1.0)
    rho = torch.logspace(0, torch.log10(d), col, device=device)
    RHO, THETA = torch.meshgrid(rho, theta, indexing='xy')
    x_new = RHO * torch.cos(THETA) + cen_x -1
    y_new = RHO * torch.sin(THETA) + cen_y -1
    x_new_norm = (2 * (x_new - 1) / (col - 1)) - 1
    y_new_norm = (2 * (y_new - 1) / (row - 1)) - 1
    grid = torch.stack((x_new_norm, y_new_norm), dim=-1)
    Img_tensor = Img.clone().detach().unsqueeze(0).unsqueeze(0)  
    out = torch.nn.functional.grid_sample(Img_tensor, grid.unsqueeze(0), mode='bicubic', align_corners=True)
    out = out.squeeze()  # Remove extra dimensions
    mask = (x_new > col) | (x_new < 0) | (y_new > row) | (y_new < 0)
    out[mask] = 0
    return out

def hpf(ht, wd):
    """
    high-pass filter function
    designed for use with Fourier-Mellin stuff
    """
    device = torch.device('cpu')
    # Calculate the resolutions for height and width
    res_ht = 1 / (ht - 1)
    res_wd = 1 / (wd - 1)    
    # Create eta and neta arrays using cos function
    eta = torch.cos(torch.pi * torch.linspace(-0.5, 0.5, wd, device=device))
    neta = torch.cos(torch.pi * torch.linspace(-0.5, 0.5, ht, device=device))  
    # Create meshgrid equivalent for eta and neta
    X1, X2 = torch.meshgrid(eta, neta, indexing='xy')  
    # Compute the element-wise product
    X = X1 * X2  
    # High-pass filter
    H = (1.0 - X) * (2.0 - X)  
    return H


def estimate_global_rot_trans(I1, I2):
    '''
    the origin is on the top-left 
    positive direction of x/y axis is to the right/bottom
    positive direction of u/v (flow) axis is also to the right/bottom
    -Theta --- clockwise roatationa angle from I1 to I2
    -Tx & -Ty --- translation of x&y direction from I1 to I2
    i.e. I1 --> anticlockwise Theta --> I1' --> translation I1'(x-Tx,y-Ty) = I2
    watch out the sign
    
    Args:
        I1,I2 (torch.Tensor): 2D tensor of shape (H, W).
    Returns:
        Theta, -Tx, -Ty
    '''
    # Define the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move I1 and I2 to the device
    I1 = I1.to(device)
    I2 = I2.to(device)
    # Get sizes of the images
    SizeR, SizeC = I1.shape
    # Compute FFTs of both images
    FA = torch.fft.fftshift(torch.fft.fft2(I1))
    FB = torch.fft.fftshift(torch.fft.fft2(I2))
    # Apply high-pass filter to the magnitude of the FFTs
    IA = hpf(SizeR, SizeC).to(device) * torch.abs(FA)
    IB = hpf(SizeR, SizeC).to(device) * torch.abs(FB)
    # Transform the high-passed FFT phase to log-polar space
    L1 = polar_mag_ft(IA).to(device)
    L2 = polar_mag_ft(IB).to(device)
    # Convert log-polar magnitude spectra to FFT
    THETA_F1 = torch.fft.fft2(L1)
    THETA_F2 = torch.fft.fft2(L2)
    # Compute cross power spectrum of F1 and F2
    a1 = torch.angle(THETA_F1)
    a2 = torch.angle(THETA_F2)
    THETA_CROSS = torch.exp(1j * (a1 - a2))
    THETA_PHASE = torch.real(torch.fft.ifft2(THETA_CROSS))
    # Find the peak of the phase correlation
    THETA_tmp_ = torch.argmax(THETA_PHASE)
    #Convert the flattened index to 2D row and column indices
    THETA_tmp, _ = divmod(THETA_tmp_.item(), THETA_PHASE.size(1))
    THETA_tmp = THETA_tmp + 1 # 0-based indices
    # Compute angle of rotation
    DPP = 360 / THETA_PHASE.shape[1]
    Theta = DPP * (THETA_tmp - 1)
    # Rotate image back by theta and theta + 180
    # Rotate the image by degrees counterclockwise
    R1 = TF.rotate(I2.unsqueeze(0), -Theta, interpolation=TF.InterpolationMode.BILINEAR).squeeze().to(device)
    R2 = TF.rotate(I2.unsqueeze(0), -(Theta + 180), interpolation=TF.InterpolationMode.BILINEAR).squeeze().to(device)
    # Compute cross power spectrum of R1_F2 and F2
    R1_F2 = torch.fft.fftshift(torch.fft.fft2(R1))
    a1 = torch.angle(FA)
    a2 = torch.angle(R1_F2)
    R1_F2_CROSS = torch.exp(1j * (a1 - a2))
    R1_F2_PHASE = torch.real(torch.fft.ifft2(R1_F2_CROSS))
    # Compute cross power spectrum of R2_F2 and F2
    R2_F2 = torch.fft.fftshift(torch.fft.fft2(R2))
    a1 = torch.angle(FA)
    a2 = torch.angle(R2_F2)
    R2_F2_CROSS = torch.exp(1j * (a1 - a2))
    R2_F2_PHASE = torch.real(torch.fft.ifft2(R2_F2_CROSS))
    # Decide whether to flip 180 or not based on correlation
    MAX_R1_F2 = torch.max(R1_F2_PHASE)
    MAX_R2_F2 = torch.max(R2_F2_PHASE)
    if MAX_R1_F2 > MAX_R2_F2:
        y, x = torch.nonzero(R1_F2_PHASE == torch.max(R1_F2_PHASE))[0] + 1 # 0-based indices, plus 1
        R = R1
    else:
        y, x = torch.nonzero(R2_F2_PHASE == torch.max(R2_F2_PHASE))[0] + 1
        if Theta < 180:
            Theta = Theta + 180
        else:
            Theta = Theta - 180
        R = R2
    # Ensure correct translation by taking from correct edge
    Tx = x.item() - 1
    Ty = y.item() - 1
    if Tx > SizeC / 2:
        Tx = Tx - SizeC
    if Ty > SizeR / 2:
        Ty = Ty - SizeR    
    return -Theta, -Tx, -Ty

def flow_by_rot_trans(alpha_, shift_x_, shift_y_, row, col, device0=None):
    """
    Calculate optical flow from I1 to I2 by estimated global rotation and translation.
    
    Parameters:
    alpha_ : rotation angle in degrees (clockwise, positive)
    shift_x_ : translation along the x-axis (negative if opposite direction)
    shift_y_ : translation along the y-axis (negative if opposite direction)
    row, col : size of the image (rows, columns)
    positive direction of x/y axis is to the right/bottom
    positive direction of u/v (flow) axis is also to the right/bottom

    Returns:
    u : Optical flow in the x direction
    v : Optical flow in the y direction
    """  
    
    if device0 is None:
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Adjust signs for rotation and translation to match the function behavior
    if isinstance(alpha_, torch.Tensor):
        alpha = alpha_.clone().detach().to(dtype=torch.float32, device=device0)
    else:
        alpha = torch.tensor(alpha_, dtype=torch.float32, device=device0)
    if isinstance(shift_x_, torch.Tensor):
        shift_x = shift_x_.clone().detach().to(dtype=torch.float32, device=device0)
    else:
        shift_x = torch.tensor(shift_x_, dtype=torch.float32, device=device0)    
    if isinstance(shift_y_, torch.Tensor):
        shift_y = shift_y_.clone().detach().to(dtype=torch.float32, device=device0)
    else:
        shift_y = torch.tensor(shift_y_, dtype=torch.float32, device=device0)      
    # Create meshgrid of coordinates
    x, y = torch.meshgrid(torch.arange(col, dtype=torch.float32, device=device0)+1, torch.arange(row, dtype=torch.float32, device=device0)+1,indexing='xy')   
    # Calculate the center point (origin of rotation)
    x0 = (1 + col) / 2
    y0 = (1 + row) / 2
    # Calculate the rotated coordinates
    x_r = (x - x0) * torch.cos(torch.deg2rad(alpha)) - (y - y0) * torch.sin(torch.deg2rad(alpha)) + x0
    y_r = (x - x0) * torch.sin(torch.deg2rad(alpha)) + (y - y0) * torch.cos(torch.deg2rad(alpha)) + y0
    # Calculate the optical flow components from the rotation
    u_r = x_r - x
    v_r = y_r - y
    # Add the translational component
    u = u_r + shift_x
    v = v_r + shift_y
    return u, v

class HuberL1Params:
    def __init__(self, 
                 warps=5, 
                 lamda=0.01, 
                 theta=0.01,
                 epsilon=0.04, 
                 max_iteration=10, 
                 nscales=3, 
                 sigma=0.8, 
                 zfactor=0.65, 
                 gscale=255, 
                 global_rot_tran = (0,0,0)):
        '''
        # tau,            // time step
        # lambda,         // weight parameter for the data term
        # warps,          // number of warpings per scale
        # theta,          // weight parameter for (u - v)²
        # nscales,        // number of scales
        # zfactor,        // zoom factor, down sampling scale parameter
        # sigma,          // standard deviation of Guassian kernel
        # epsilon,        // huber norm
        # global_rot_tran // (alpha, shift_x, shift_y)  rotation degree, clockwise, unit: ° , translation along two axises
        '''
        self.warps = warps
        self.lamda = lamda
        self.theta = theta
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.tau = 1 / (4 + epsilon)
        self.nscales = nscales
        self.sigma = sigma
        self.zfactor = zfactor
        self.gscale = gscale
        self.global_rot_tran = global_rot_tran
        
def huber_l1_optical_flow(I1, I2, u, v, params):
    # Implementation of the Anisotropic Huber-L1 Optical Flow method
    # [1] Werlberger, Manuel, et al. "Anisotropic Huber-L1 Optical Flow." BMVC. Vol. 1. No. 2. 2009.
    # I1,             // source image
    # I2,             // target image
    # tau,            // step-width
    # lamda,         // weight parameter for the data term
    # warps,          // number of warpings per scale
    # theta,          // weight parameter for (u - v)²
    # epsilon,        // huber norm
    # u, v: initial flow fields
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = I1.device
    # Move images to the correct device
    I1 = I1.to(device)
    I2 = I2.to(device)
    u = u.to(device)
    v = v.to(device)
    # tau, lambda, theta, epsilon values from params
    warps = params.warps
    lamda = params.lamda
    theta = params.theta
    epsilon = params.epsilon
    tau = params.tau
    max_iteration = params.max_iteration
    
    H, W = I2.shape
    X, Y = torch.meshgrid(torch.linspace(0, W-1, W, device=device),torch.linspace(0, H-1, H, device=device),indexing='xy')
    #  initialization of vectors p1 & p2
    p1x = torch.zeros(H, W, device=device)
    p1y = torch.zeros(H, W, device=device)
    p2x = torch.zeros(H, W, device=device)
    p2y = torch.zeros(H, W, device=device)
    # compute central derivatives (compute the gradient with centered differences)
    I2x, I2y = compute_2d_derivatives(I2)
    u_ = torch.zeros(H, W, device=device)
    v_ = torch.zeros(H, W, device=device)
    # huber_l1_optical_flow
    for n in range(warps):
        # compute the warping of the target image and its derivatives
        # warping of I2, I2x, I2y
        I2w = warp_2d(I2, u, v) 
        I2wx = warp_2d(I2x, u, v)
        I2wy = warp_2d(I2y, u, v)
        # store the |Grad(I2)|^2
        grad2 = torch.max(torch.tensor(1e-6), I2wx**2 + I2wy**2)
        # compute the constant part of the rho function
        rho_c = I2w - I2wx * u - I2wy * v - I1
        # inner iterations 
        for r in range(max_iteration):
            # solve u_, v_ (thresholding opterator TH)
            rho = rho_c + I2wx * u + I2wy * v
            ind1 = rho < -lamda * theta * grad2
            ind2 = rho > lamda * theta * grad2
            ind3 = abs(rho) <= lamda * theta * grad2
            u_[ind1] = u[ind1] + lamda * theta * I2wx[ind1]
            v_[ind1] = v[ind1] + lamda * theta * I2wy[ind1]
            u_[ind2] = u[ind2] - lamda * theta * I2wx[ind2]
            v_[ind2] = v[ind2] - lamda * theta * I2wy[ind2]
            u_[ind3] = u[ind3] - rho[ind3] * I2wx[ind3] / grad2[ind3]
            v_[ind3] = v[ind3] - rho[ind3] * I2wy[ind3] / grad2[ind3]
            # Create padded versions of the tensors by adding a row or column of zeros
            p1x_padded = torch.cat([torch.zeros(H, 1, device=device), p1x[:, :-1]], dim=1).to(device)  # Shift p1x right and pad zeros
            p1y_padded = torch.cat([torch.zeros(1, W, device=device), p1y[:-1, :]], dim=0).to(device)  # Shift p1y down and pad zeros
            p2x_padded = torch.cat([torch.zeros(H, 1, device=device), p2x[:, :-1]], dim=1).to(device)  # Shift p2x right and pad zeros
            p2y_padded = torch.cat([torch.zeros(1, W, device=device), p2y[:-1, :]], dim=0).to(device)  # Shift p2y down and pad zeros
            # compute the divergence of the dual variable (p1, p2) by backward diferences
            div_p1 = p1x - p1x_padded + p1y - p1y_padded
            div_p2 = p2x - p2x_padded + p2y - p2y_padded
            # estimate the values of the optical flow (u, v)
            u = u_ + theta * div_p1
            v = v_ + theta * div_p2
            id1 = (X + u < 0) | (X + u > W-1)
            id2 = (Y + v < 0) | (Y + v > H-1)
            u[id1] = 0
            v[id2] = 0
            # compute the forward gradient of the optical flow 
            u_dx = torch.cat([u[:, 1:], u[:,-1].unsqueeze(1)], dim=1).to(device) - u
            u_dy = torch.cat([u[1:, :], u[-1,:].unsqueeze(0)], dim=0).to(device) - u
            v_dx = torch.cat([v[:, 1:], v[:,-1].unsqueeze(1)], dim=1).to(device) - v
            v_dy = torch.cat([v[1:, :], v[-1,:].unsqueeze(0)], dim=0).to(device) - v
            # estimate the values of the dual variable (p1, p2)
            ng1 = torch.maximum(torch.ones_like(p1x), torch.abs(p1x+tau*(u_dx-epsilon*p1x)) + torch.abs(p1y+tau*(u_dy-epsilon*p1y)))
            p1x = (p1x+tau*(u_dx-epsilon*p1x)) / ng1
            p1y = (p1y+tau*(u_dy-epsilon*p1y)) / ng1
            ng2 = torch.maximum(torch.ones_like(p2x),torch.abs(p2x+tau*(v_dx-epsilon*p2x)) + torch.abs(p2y+tau*(v_dy-epsilon*p2y)))
            p2x = (p2x+tau*(v_dx-epsilon*p2x)) / ng2
            p2y = (p2y+tau*(v_dy-epsilon*p2y)) / ng2
    return u,v


def pyramidal_huber_l1_optical_flow(I1, I2, params, verbose=False):
    # Implementation of the Anisotropic Huber-L1 Optical Flow method
    # [1] Werlberger, Manuel, et al. "Anisotropic Huber-L1 Optical Flow." BMVC. Vol. 1. No. 2. 2009.
    # I1,             // source image
    # I2,             // target image
    # tau,            // step-width
    # lamda,         // weight parameter for the data term
    # warps,          // number of warpings per scale
    # theta,          // weight parameter for (u - v)²
    # epsilon,        // huber norm
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = I1.device
    # tau, lambda, theta, epsilon values from params
    warps = params.warps
    lamda = params.lamda
    theta = params.theta
    epsilon = params.epsilon
    tau = params.tau
    max_iteration = params.max_iteration
    nscales = params.nscales
    sigma = params.sigma
    zfactor = params.zfactor
    gscale = params.gscale
    global_rot_tran = params.global_rot_tran
    alpha = global_rot_tran[0]  # Rotation in degrees
    shift_x = global_rot_tran[1]  # Shift along x-axis
    shift_y = global_rot_tran[2]  # Shift along y-axis
    # Move images to the correct device
    I1 = I1.to(device)
    I2 = I2.to(device)
    
    u = torch.zeros_like(I2, device=device)
    v = torch.zeros_like(I2, device=device) 

    I1 = norm_3d(I1.unsqueeze(0)).squeeze() * gscale
    I2 = norm_3d(I2.unsqueeze(0)).squeeze() * gscale

    I1 = gaussian_smoothing_2d(I1, sigma = sigma).to(device)
    I2 = gaussian_smoothing_2d(I2, sigma = sigma).to(device)
    # create the scales

    I1_pyramial = [None for _ in range(nscales)]
    I2_pyramial = [None for _ in range(nscales)]
    for s in range(nscales):
        if s == 0:
            I1_pyramial[s] = I1
            I2_pyramial[s] = I2
        else:
            I1_blur = gaussian_smoothing_2d(I1_pyramial[s - 1],  sigma = sigma).to(device)
            I2_blur = gaussian_smoothing_2d(I2_pyramial[s - 1],  sigma = sigma).to(device)
            # Downsample
            I1_resized = resize_2d_tensor(I1_blur, scale_factor=zfactor).to(device)
            I2_resized = resize_2d_tensor(I2_blur, scale_factor=zfactor).to(device)
            I1_pyramial[s] = I1_resized
            I2_pyramial[s] = I2_resized

    # initialize the flow
    flow_u = [None for _ in range(nscales)]
    flow_v = [None for _ in range(nscales)] 
    # Retrieve global rotation and translation values
    ratio1 = I1.shape[0] / I1_pyramial[nscales - 1].shape[0]  # Height ratio
    ratio2 = I1.shape[1] / I1_pyramial[nscales - 1].shape[1]  # Width ratio
    # Define flow initialization function, assuming flow_by_rot_trans is defined elsewhere
    flow_u[nscales - 1], flow_v[nscales - 1] = flow_by_rot_trans(
        global_rot_tran[0],                       # Rotation
        global_rot_tran[1] / ratio2,              # Translated rotation adjusted by width ratio
        global_rot_tran[2] / ratio1,              # Translated rotation adjusted by height ratio
        I1_pyramial[nscales - 1].shape[0],        # Height
        I1_pyramial[nscales - 1].shape[1]         # Width
    )
    flow_u[nscales - 1] = flow_u[nscales - 1].to(device)
    flow_v[nscales - 1] = flow_v[nscales - 1].to(device)
    # pyramidal approximation to the optic flow
    # print("Pyramidal huber_l1 scale: ", end="")
    # Add the conditional tqdm logic
    if verbose:
        print("Pyramidal huber_l1 scale: ", end="")
    for s in range(nscales - 1, -1, -1):
        # compute the optical flow at this scale 
        flow_u[s], flow_v[s] = huber_l1_optical_flow(I1_pyramial[s], I2_pyramial[s], flow_u[s], flow_v[s], params)
        # zoom (upsampling) the optic flow for the next finer scale then scale the optic flow 
        if s != 0:
            flow_u[s - 1] = 1 / zfactor * resize_2d_tensor(flow_u[s], size=I1_pyramial[s-1].shape).to(device)
            flow_v[s - 1] = 1 / zfactor * resize_2d_tensor(flow_v[s], size=I2_pyramial[s-1].shape).to(device)
            flow_u[s - 1] = median_filter_2d(flow_u[s - 1]).to(device)
            flow_v[s - 1] = median_filter_2d(flow_v[s - 1]).to(device)
        if verbose:
            print(f"{s+1}-", end="")
    if verbose:
        print("\n")
    # Final output at the finest scale
    U = flow_u[0]
    V = flow_v[0]
    return U,V
