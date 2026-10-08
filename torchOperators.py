#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:51:33 2024

@author: bar
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import griddata
#from scipy.interpolate import LinearNDInterpolator
import matplotlib.tri as tri
from torch import Tensor
import mapbox_earcut as earcut   # C++ ear-clipping, tolerant & fast
#%%
class gridInterpolator:
    def __init__(self, X, Y, values, device='cpu',dataType=None):
        """
        Initialize the interpolator with grid X, Y, and Values.
        
        Args:
        - X: 2D tensor or NumPy array for the X-coordinates of the grid.
        - Y: 2D tensor or NumPy array for the Y-coordinates of the grid.
        - Values: 2D tensor or NumPy array for the values at each grid point.
        - device: The device where computations will take place ('cpu' or 'cuda').
        """
        # Set device
        self.device = device
        self.dataType=dataType
        if self.dataType is None:
            self.dataType=torch.float32

        # Convert inputs to tensors and move them to the device
        self.X = self._convert_to_tensor(X).to(device)
        self.Y = self._convert_to_tensor(Y).to(device)
        self.values = self._convert_to_tensor(values).to(device)

        # Normalize X and Y to the range [-1, 1]
        self.x_min, self.x_max = self.X.min(), self.X.max()
        self.y_min, self.y_max = self.Y.min(), self.Y.max()
        
        # Prepare grid values for grid_sample
        self.grid_values = self.values.unsqueeze(0).unsqueeze(0).to(device)  # Shape (1, 1, H, W)


    def _convert_to_tensor(self, data):
        """
        Converts input data to a PyTorch tensor with the specified data type.
    
        Args:
        - data: Input data (torch.Tensor, numpy.ndarray, or list).
    
        Returns:
        - torch.Tensor: The converted tensor with the specified data type.
    
        Raises:
        - TypeError: If the input data is not a valid type or cannot be converted to the desired data type.
        """
        if isinstance(data, torch.Tensor):
            if data.dtype != self.dataType:
                # Convert to the desired data type
                return data.to(dtype=self.dataType)
            return data
        elif isinstance(data, (np.ndarray, list)):
            # Convert NumPy array or list to tensor with the desired data type
            return torch.tensor(data, dtype=self.dataType)
        else:
            raise TypeError("Input must be a torch.Tensor, numpy.ndarray, or list.")

    def _normalize_points(self, points):
        """
        Normalize points to the range [-1, 1].
        """
        #points = points.to(self.device)  # Ensure points are on the correct device
        x_normalized = 2 * (points[:, 0] - self.x_min) / (self.x_max - self.x_min) - 1
        y_normalized = 2 * (points[:, 1] - self.y_min) / (self.y_max - self.y_min) - 1
        return torch.stack((x_normalized, y_normalized), dim=1)
    
    def Query(self, scatter_points):
        """
        Query interpolated values at scatter points.
        
        Args:
        - scatter_points: Tensor of shape (N, 2) with points in the original coordinate system.
        
        Returns:
        - Tensor of interpolated values at the query points.
        """
        scatter_points = self._normalize_points(scatter_points)  # Normalize to [-1, 1]
        scatter_points = scatter_points.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, N, 2)
        
        interpolated_values = F.grid_sample(
            self.grid_values,
            scatter_points,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        return interpolated_values.squeeze()  # Return as 1D tensor for N points
 
    def normalize_to_pdf(self):
        """
        Normalize the grid values so that they sum up to 1.
        This converts the grid values to a probability density function (PDF).
        """
        total_sum = self.grid_values.sum()
        if total_sum > 0:
            self.grid_values = self.grid_values / total_sum
        else:
            raise ValueError("The sum of grid values is zero or negative, normalization is not possible.")
            
    def FindMaxValue(self):
        return torch.max(self.values[~torch.isnan(self.values)])
    
    
    def with_nan_to_zero(self):
        """
        Return a new gridInterpolator instance with all NaN values in the grid replaced by zero.
        
        This method leaves the original instance unchanged.
        """
        # Replace NaNs with zeros in the 'values' tensor.
        new_values = torch.nan_to_num(self.values, nan=0.0)
        # Create and return a new instance with the updated values.
        return gridInterpolator(self.X, self.Y, new_values, device=self.device, dataType=self.dataType)
    
    
#%%
class rejectionSampler:
    def __init__(self,interpolator):
        self.interpolator=interpolator
        self.device=interpolator.device
        self.dtype=interpolator.dataType
        self.maxValue=self.interpolator.FindMaxValue()
        
    def VectorOfUnifromPoints(self,N=1024,xmin=0,xmax=1):
        return torch.empty(N, device=self.device, dtype=self.dtype).uniform_(xmin, xmax)
    
    def GenerateRandomPointsWithDomain(self,N=1024):
              
        x_values = self.VectorOfUnifromPoints(N,self.interpolator.x_min,self.interpolator.x_max)
        y_values = self.VectorOfUnifromPoints(N,self.interpolator.y_min,self.interpolator.y_max)
        return torch.stack((x_values, y_values), dim=1)
    
    def Sample(self,N):
        
        acceptedPoints = torch.empty((0, 2), device=self.device)
        
        while len(acceptedPoints) < N:

            points=self.GenerateRandomPointsWithDomain(2*N)
            randomNumbers=self.VectorOfUnifromPoints(2*N,0,self.maxValue)
            valueOfPoints=self.interpolator.Query(points)
            
            maskAccept=valueOfPoints>randomNumbers
            acceptedPoints = torch.cat([acceptedPoints, points[maskAccept]])
        
        
        return acceptedPoints[:N]
        
#%%
class gridBasedDistribution:
    def __init__(self, X, Y, values, device='cpu',dataType=None):
        self.device = device
        self.dataType=dataType
        if self.dataType is None:
            self.dataType=torch.float32

            
        self.interpolator=gridInterpolator(X,Y,values,device,self.dataType)
        self.recjectionSampler=rejectionSampler(self.interpolator)
        
        self.normalizeExtent=torch.tensor([self.interpolator.x_min,self.interpolator.x_max,self.interpolator.y_min,self.interpolator.y_max],device=self.device, dtype=self.dataType)
    
    def Sample(self,N,normalize=True):
        if normalize:
            return self.normalize_points(self.recjectionSampler.Sample(N))
        else:
            return self.recjectionSampler.Sample(N)
    
    def SampleValue(self,points,denormalize_points=True):
        if denormalize_points:
            return self.interpolator.Query(self.denormalize_points(points))
        else:
            return self.interpolator.Query(points)
    
    def normalize_points(self,points):
        """
        Normalize points between 0 and 1 for both x and y axes.
    
        Args:
        - points (torch.Tensor): A tensor of shape (N, 2) containing points [x, y].
        - xmin, xmax (float): Minimum and maximum bounds for the x-axis.
        - ymin, ymax (float): Minimum and maximum bounds for the y-axis.
    
        Returns:
        - torch.Tensor: A tensor of shape (N, 2) with normalized points.
        """
        # Normalize x and y axes separately
        xmin=self.normalizeExtent[0];xmax=self.normalizeExtent[1];ymin=self.normalizeExtent[2];ymax=self.normalizeExtent[3]
        normalized_x = (points[:, 0] - xmin) / (xmax - xmin)
        normalized_y = (points[:, 1] - ymin) / (ymax - ymin)
        return torch.stack((normalized_x, normalized_y), dim=1)
    
    def denormalize_points(self,points):
        """
        Denormalize points from [0, 1] back to the original coordinate space.
    
        Args:
        - points (torch.Tensor): A tensor of shape (N, 2) containing normalized points [x, y].
        - xmin, xmax (float): Original minimum and maximum bounds for the x-axis.
        - ymin, ymax (float): Original minimum and maximum bounds for the y-axis.
    
        Returns:
        - torch.Tensor: A tensor of shape (N, 2) with denormalized points.
        """
        
        xmin=self.normalizeExtent[0];xmax=self.normalizeExtent[1];ymin=self.normalizeExtent[2];ymax=self.normalizeExtent[3]
        denormalized_x = points[:, 0] * (xmax - xmin) + xmin
        denormalized_y = points[:, 1] * (ymax - ymin) + ymin
        return torch.stack((denormalized_x, denormalized_y), dim=1)
    
    
    def with_nan_to_zero(self):
        """
        Return a new gridBasedDistribution instance with all NaN values in the underlying grid values replaced by zero.
        
        This method performs the conversion separately from the interpolator, replacing NaNs in the 'values' tensor.
        It then creates a new instance using the same X and Y values, but with the modified grid values.
        """
        # Replace NaNs with zeros in the grid values.
        new_values = torch.nan_to_num(self.interpolator.values, nan=0.0)
        # Create and return a new gridBasedDistribution instance using the updated values.
        return gridBasedDistribution(self.interpolator.X,
                                     self.interpolator.Y,
                                     new_values,
                                     device=self.device,
                                     dataType=self.dataType)
    
    def ReturnCouplingSampler(self,device):
        """ this is a temp function """
        return gridBasedDistribution(self.interpolator.X,
                                     self.interpolator.Y,
                                     self.interpolator.values,
                                     device=device,
                                     dataType=self.dataType)
    
    
    
    def CorrectinterpolatorBetween_0_1(self):
        self.interpolator.values.clamp_(0.0, 1.0)
        
#%%

class TransformedGridBasedDistribution:
    def __init__(self, base_dist, angle_deg=0.0, dx=0.0, dy=0.0, pivot=None):
        self.base = base_dist
        self.device = base_dist.device
        self.dtype = base_dist.dataType

        self.angle_deg = float(angle_deg)
        self.dx = float(dx)
        self.dy = float(dy)

        # pivot in physical coords (x0,y0). If None use center of base grid.
        if pivot is None:
            X = base_dist.interpolator.X
            Y = base_dist.interpolator.Y
            x0 = 0.5 * (X.min() + X.max())
            y0 = 0.5 * (Y.min() + Y.max())
            pivot = (x0, y0)
        self.px, self.py = map(float, pivot)
    
    
    def set_angle_dx_dy(self, angle_deg=0.0, dx=0.0, dy=0.0):
        self.angle_deg = float(angle_deg)
        self.dx = float(dx)
        self.dy = float(dy)

    def _inverse_transform(self, pts):
        # pts: torch (N,2) in rotated coords
        th = torch.tensor(-np.deg2rad(self.angle_deg), device=pts.device, dtype=pts.dtype)
        c, s = torch.cos(th), torch.sin(th)

        x = pts[:,0] - self.px - self.dx
        y = pts[:,1] - self.py - self.dy

        xb =  c*x - s*y + self.px
        yb =  s*x + c*y + self.py
        return torch.stack([xb, yb], dim=1)
    
    
    def RotateAndTranslate(self,pts):
        th = torch.tensor(np.deg2rad(self.angle_deg), device=pts.device, dtype=pts.dtype)
        c, s = torch.cos(th), torch.sin(th)
        x = pts[:,0] - self.px
        y = pts[:,1] - self.py
        xr = c*x - s*y + self.px + self.dx
        yr = s*x + c*y + self.py + self.dy
        return torch.stack([xr, yr], dim=1)
    
    def set_pivot(self, pivot):
        """
        Update pivot in-place.
    
        pivot can be:
          - tuple/list/np.ndarray/torch.Tensor with (x0,y0)
        """
        if isinstance(pivot, torch.Tensor):
            p = pivot.to(device=self.device, dtype=self.dtype).flatten()
            if p.numel() != 2:
                raise ValueError("pivot tensor must have 2 elements (x0,y0).")
            x0, y0 = float(p[0].item()), float(p[1].item())
        else:
            x0, y0 = map(float, pivot)
    
        self.pivot = (x0, y0)
        self.px, self.py = map(float, pivot)

    def Sample(self, N, normalize=False):
        # easiest: sample from base and forward-transform points (not shown),
        # OR keep sampling in rotated bbox and evaluate base via inverse.
        # For now: sample base then forward transform so you always get N points quickly.
        pts = self.base.Sample(N, normalize=False)
        pts_r=self.RotateAndTranslate(pts)
        return self.base.normalize_points(pts_r) if normalize else pts_r

    def SampleValue(self, points, denormalize_points=True):
        # points can be normalized or not, reuse base helpers
        pts_base = self._inverse_transform(points)
        pts = self.base.denormalize_points(pts_base) if denormalize_points else pts_base
        return self.base.interpolator.Query(pts)   

        
        
#%%  

def LoadManyGridDisurbtionsWithBaseGrid(basegrid,values):
    if basegrid.interpolator.X.numpy().shape != values.shape[:2]:
        raise TypeError("base and new values do no correlate")
        
    manygrids=[]
    X=basegrid.interpolator.X.numpy();Y=basegrid.interpolator.Y.numpy()
    
    for i in range(values.shape[-1]):
        manygrids.append(gridBasedDistribution(X, Y, values[:,:,i]))
        
    return manygrids
   

#%% 
# class EQGridBasedDistribution(gridBasedDistribution):
#     def __init__(self, X, Y, values, device='cpu', dataType=None, slipForm2=None,eventDict=None):
#         # Call the parent class's __init__ method
#         super().__init__(X, Y, values, device, dataType)
#         self.slipForm2 = slipForm2
#         self.eventDict = eventDict

#%%
def generateGridBaseDistributionFromScatter(x,y, values,device='cpu',dataType=None, grid_size=(100, 100), method='cubic'):

    """
    Interpolates scattered points with values onto a regular grid.

    Args:
        x (np.ndarray): N array of x coordinates.
        y (np.ndarray): N array of y coordinates.
        values (np.ndarray): N array of values at the points.
        grid_size (tuple): Size of the grid (num_x, num_y).
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        grid_x, grid_y, grid_values (np.ndarray): Gridded results.
    """
    # Compute bounds
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Create grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size[0]),
        np.linspace(y_min, y_max, grid_size[1])
    )

    # Interpolate values to grid
    grid_values = griddata((x, y), values, (grid_x, grid_y), method=method)
    grid_values = np.nan_to_num(grid_values, nan=0.0)
    

    return gridBasedDistribution(grid_x,grid_y,grid_values,device,dataType)


#%%

def generate_triangular_grid_from_triangulation(triangulation,values,grid_size=(100, 100),device='cpu',dataType=None):
    """
    Interpolates per-triangle values onto a regular grid using a precomputed Triangulation.

    Args:
        triangulation (matplotlib.tri.Triangulation): Precomputed triangulation object.
        values (np.ndarray): 1D array of values corresponding to each triangle.
        grid_size (tuple, optional): Size of the grid (num_x, num_y). Default is (100, 100).

    Returns:
        grid_x (np.ndarray): 2D array of x-coordinates of the grid.
        grid_y (np.ndarray): 2D array of y-coordinates of the grid.
        grid_values (np.ndarray): 2D array of interpolated values on the grid.
    """
    
    # Validate inputs
    if not isinstance(triangulation, tri.Triangulation):
        raise TypeError("triangulation must be an instance of matplotlib.tri.Triangulation")
    
    num_triangles = len(triangulation.triangles)
    if len(values) != num_triangles:
        raise ValueError(
            f"Length of values ({len(values)}) must match the number of triangles ({num_triangles}) in the triangulation"
        )
    
    # Compute bounds from triangulation vertices
    x_min, x_max = triangulation.x.min(), triangulation.x.max()
    y_min, y_max = triangulation.y.min(), triangulation.y.max()
    
    # Create a regular grid within the bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size[0]),
        np.linspace(y_min, y_max, grid_size[1])
    )
    values=np.array(values)
    # Initialize grid_values with NaNs
    grid_values = np.full(grid_x.shape, 0)
    
    # Create a TriFinder object for efficient triangle lookup
    trifinder = triangulation.get_trifinder()
    
    # Flatten the grid points for vectorized processing
    flat_grid_x = grid_x.flatten()
    flat_grid_y = grid_y.flatten()
    
    # Find which triangle each grid point belongs to
    simplex = trifinder(flat_grid_x, flat_grid_y)
    
    # Identify valid grid points that fall within a triangle
    valid = simplex != -1
    
    # Ensure 'simplex' contains integer indices
    simplex = simplex.astype(int)
    
    # Assign triangle values to grid points where simplex is valid
    grid_values_flat = np.full_like(flat_grid_x, 0)
    grid_values_flat[valid] = values[simplex[valid]]
    
    # Reshape to grid shape
    grid_values = grid_values_flat.reshape(grid_x.shape)
    

    return gridBasedDistribution(grid_x, grid_y, grid_values, device, dataType)

            
#%%
def MoveSamples(G,samples,sourceDistribution):
    return G(sourceDistribution.normalize_points(samples))


#%%



class FastUniformPolySampler(torch.nn.Module):
    """
    Uniform sampler for any simple polygon (convex *or* concave).

    Parameters
    ----------
    poly_xy : (M,2) numpy array / sequence / torch.Tensor
    device  : str or torch.device, optional
        'cpu', 'cuda', 'cuda:0', ...  (default: infer from input or CPU)
    """

    def __init__(self, poly_xy, device=None):
        super().__init__()

        # ---------- normalise input ----------
        if isinstance(poly_xy, Tensor):
            base_dev = poly_xy.device
        else:
            poly_xy = np.asarray(poly_xy, dtype=np.float32)
            base_dev = torch.device("cpu")

        device = torch.device(device) if device is not None else base_dev
        poly_xy = torch.as_tensor(poly_xy, dtype=torch.float32, device=device)

        if poly_xy.ndim != 2 or poly_xy.size(1) != 2:
            raise ValueError("poly_xy must be (M,2)")

        # ----- CCW order (earcut wants CCW for outer ring) ---------------
        if self._signed_area(poly_xy) < 0:
            poly_xy = torch.flip(poly_xy, (0,))

        self.register_buffer("poly_xy", poly_xy)

        # ---------- triangulate on CPU with earcut ----------
        tris_cpu = self._triangulate_earcut(poly_xy.cpu().numpy())
        self.register_buffer("triangles", tris_cpu.to(device))

        # ---------- area CDF (on target device) -----------
        v0, v1, v2 = (poly_xy[self.triangles[:, i]] for i in range(3))
        areas = 0.5 * torch.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        )
        self.register_buffer("areas_cdf", torch.cumsum(areas, 0))
        self.total_area = self.areas_cdf[-1]

    # -------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, n: int) -> Tensor:
        """Draw `n` uniform points inside the polygon, returned on self.device."""
        dev = self.poly_xy.device
        r = torch.rand(n, device=dev) * self.total_area
        tri_idx = torch.searchsorted(self.areas_cdf, r, right=False)

        v0 = self.poly_xy[self.triangles[tri_idx, 0]]
        v1 = self.poly_xy[self.triangles[tri_idx, 1]]
        v2 = self.poly_xy[self.triangles[tri_idx, 2]]

        u = torch.rand(n, device=dev)
        v = torch.rand(n, device=dev)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]

        return v0 + u.unsqueeze(1) * (v1 - v0) + v.unsqueeze(1) * (v2 - v0)

    # -------------------------------------------------------------------
    @staticmethod
    def _signed_area(p: Tensor):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * torch.sum(x.roll(-1) * y - y.roll(-1) * x)

    # ------------------------------------------------------------------
    @staticmethod
    def _triangulate_earcut(poly_np: np.ndarray) -> torch.Tensor:
        """
        Triangulate a single-ring polygon (no holes) with mapbox_earcut ≥ 1.0.

        Returns
        -------
        torch.Tensor  (K, 3)  -- triangle indices into the original
                                 vertex list (int64, clockwise or CCW).
        """
        import mapbox_earcut as earcut
        if poly_np.ndim != 2 or poly_np.shape[1] != 2:
            raise ValueError("poly must be (n,2)")

        verts  = poly_np.astype(np.float32)            # (n,2) for earcut
        rings  = np.array([verts.shape[0]], dtype=np.uint32)  # single ring
        idx    = earcut.triangulate_float32(verts, rings)     # uint32 1-D
        return torch.from_numpy(idx.astype(np.int64)).view(-1, 3)



     
