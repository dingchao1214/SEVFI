import os
import torch
import numpy as np
import torch.nn.functional as F

##############################################################################
### disparity warping codes credit from: https://github.com/haofeixu/aanet ###
##############################################################################
def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity (with right img and left disparity)
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask

##############################################################################
##############################################################################
def normalization(data):
    return (data - data.min())/(data.max()-data.min())

def get_filename(path, suffix):
    ## function used to get file names
    namelist=[]
    filelist = os.listdir(path)
    for i in filelist:
        if os.path.splitext(i)[1] == suffix:
            namelist.append(i)
    namelist.sort()
    return namelist

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def filter_events_by_time(x, y, p, t, start, end):
    ## filter events based on temporal dimension
    x = x[t>=start]
    y = y[t>=start]
    p = p[t>=start]
    t = t[t>=start]

    x = x[t<=end]
    y = y[t<=end]
    p = p[t<=end]
    t = t[t<=end]
    return (x, y, p, t)

def reverse_events(x, y, p, t):
    if len(t) > 0:
        p[p == 1] = 2
        p[p == 0] = 1
        p[p == 2] = 0
        t = t[-1] - t
        x_r = np.flipud(x)
        y_r = np.flipud(y)
        t_r = np.flipud(t)
        p_r = np.flipud(p)
        return x_r, y_r, p_r, t_r
    else:
        return x, y, p, t

def rectify_events(rectify_ev_maps, x: np.ndarray, y: np.ndarray, height, width):
    # From distorted to undistorted
    rectify_map = rectify_ev_maps
    assert rectify_map.shape == (height, width, 2), rectify_map.shape
    assert x.max() < width
    assert y.max() < height
    return rectify_map[y, x]

#### from DSEC
class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError

class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2*pol-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

def warp_images_with_flow(images, flow):
    """
    Generates a prediction of an image given the optical flow, as in Spatial Transformer Networks.
    """
    dim3 = 0
    if images.dim() == 3:
        dim3 = 1
        images = images.unsqueeze(0)
        flow = flow.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]
    flow_x, flow_y = flow[:, 0, ...], flow[:, 1, ...]
    coord_y, coord_x = torch.meshgrid(torch.arange(height), torch.arange(width))

    pos_x = coord_x.reshape(height, width).type(torch.float32).cuda() + flow_x
    pos_y = coord_y.reshape(height, width).type(torch.float32).cuda() + flow_y
    pos_x = (pos_x - (width - 1) / 2) / ((width - 1) / 2)
    pos_y = (pos_y - (height - 1) / 2) / ((height - 1) / 2)

    pos = torch.stack((pos_x, pos_y), 3).type(torch.float32)
    result = torch.nn.functional.grid_sample(images, pos, mode='bilinear', padding_mode='zeros')
    if dim3 == 1:
        result = result.squeeze()

    return result
