#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:30:53 2025

@author: bar
"""

import numpy as np
from pyproj import Proj
import matplotlib.pyplot as plt
import torch
import torchOperators as to
from pyproj import Geod
from scipy.interpolate import griddata
import seaborn as sns
import pandas as pd
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#%%
SLIP_PER_AREA=10
#%% helper function to compute edges of faults
def subfault_corners(xc, yc, zc, strike_deg, dip_deg, L, W, ref_loc="top_center"):
    """
    Return the four 3D corner coordinates (TL → TR → BR → BL) of a rectangular subfault.

    Parameters
    ----------
    xc, yc, zc : float
        The reference point’s x, y, z (in the local coordinate system).
    strike_deg : float
        Strike angle in degrees.
    dip_deg : float
        Dip angle in degrees.
    L, W : float
        Along‐strike length (L) and down‐dip width (W) of the patch.
    ref_loc : str, optional (default="top_center")
        Which point (xc, yc, zc) refers to:
          - "top_center":    the patch’s midpoint along strike, at the top edge.
          - "top_left":      the upper‐left corner of that top edge.
          - "top_right":     the upper‐right corner of that top edge.
          - "center":        the geographic center (centroid) of the patch.

    Returns
    -------
    np.ndarray of shape (4, 3)
        The four corners in this order: TL, TR, BR, BL.
    """
    # Convert degrees → radians
    strike = np.deg2rad(strike_deg)
    dip    = np.deg2rad(dip_deg)

    # Unit vectors
    s_hat = np.array([ np.sin(strike),   np.cos(strike),   0.0 ])  # along‐strike
    #h_hat = np.array([-np.cos(strike),   np.sin(strike),   0.0 ])  # horizontal down‐dip - old version ?
    h_hat = np.array([ np.cos(strike), -np.sin(strike), 0.0 ])
    d_hat = h_hat * np.cos(dip) + np.array([0.0, 0.0, -np.sin(dip)])  # full down‐dip

    half_len = 0.5 * L

    # Determine the top‐center location from the chosen reference
    if ref_loc == "top_center":
        top_ctr = np.array([xc, yc, zc])

    elif ref_loc == "top_left":
        # Given point is TL; move half_len along +s_hat to get top‐center
        top_ctr = np.array([xc, yc, zc]) + half_len * s_hat

    elif ref_loc == "top_right":
        # Given point is TR; move half_len along –s_hat to get top‐center
        top_ctr = np.array([xc, yc, zc]) - half_len * s_hat

    elif ref_loc == "center":
        # Given point is centroid; move (W/2) up‐dip (i.e., subtract (W/2)*d_hat)
        top_ctr = np.array([xc, yc, zc]) - 0.5 * W * d_hat

    else:
        raise ValueError("ref_loc must be 'top_center', 'top_left', 'top_right', or 'center'")

    # Now compute the four corners from top‐center
    TL = top_ctr - half_len * s_hat
    TR = top_ctr + half_len * s_hat
    BL = TL + W * d_hat
    BR = TR + W * d_hat

    return np.vstack([TL, TR, BR, BL])

#%%
class FaultSegment:
    """
    Base class: holds slip & geometric info common to all segments.
    Subclasses must provide `.coords_x` and `.coords_y` properties (2-D arrays).
    """
    def __init__(self,
                 slip:   np.ndarray,
                 x:      np.ndarray,
                 y:      np.ndarray,
                 z: np.ndarray,
                 dip:    float,
                 rake:   float,
                 strike: float,
                 DimWL: np.ndarray,
                 position: str =  "top_center",
                 additionalData: pd.DataFrame = pd.DataFrame({})
                 ):
        
        self.slip   = np.atleast_1d(slip)
        self.dip    = dip
        self.rake   = rake
        self.strike = strike
        self.x    = np.atleast_1d(x)
        self.y    = np.atleast_1d(y)
        self.DimWL    = np.atleast_1d(DimWL)
        self.z    = np.atleast_1d(z)
        self.position=position
        self.additionalData=additionalData
        assert self.x.shape == self.y.shape == self.slip.shape, \
            f"Shape mismatch – xs: {self.x.shape}, ys: {self.y.shape}, slip: {self.slip.shape}"
        
        if self.additionalData is not None and not self.additionalData.empty:
            assert len(self.additionalData) == len(self.x), \
                f"Length mismatch – additionalData: {len(self.additionalData)}, x: {len(self.x)}"

        
    @property
    def max_slip(self) -> float:
        return float(np.nanmax(self.slip))
    
    @property
    def extent(self) -> tuple[float, float, float, float]:
        """
        Return the bounding box of this segment in degrees:
           (lon_min, lon_max, lat_min, lat_max)
        """
        return (
            float(self.x.min()), 
            float(self.x.max()), 
            float(self.y.min()), 
            float(self.y.max())
        )
    
    @property
    def coords_x(self) -> np.ndarray:
        return self.x

    @property
    def coords_y(self) -> np.ndarray:
        return self.y

    def _spacing(self, arr, axis: int, dim_idx: int) -> float:
        """Compute mean delta along `axis`, or fallback to self.DimWL[dim_idx]."""
        a = np.atleast_2d(arr)
        if a.shape[axis] > 1:
            return float(np.nanmean(np.diff(a, axis=axis)))
        if getattr(self, "DimWL", None) is not None:
            return float(self.DimWL[dim_idx])
        raise ValueError(f"Cannot compute spacing along axis {axis}")

    @property
    def dx(self) -> float:
        return self._spacing(self.x, axis=1, dim_idx=0)

    @property
    def dy(self) -> float:
        return self._spacing(self.y, axis=0, dim_idx=1)


    def PlotSlip(self,ax=None):
        if ax is None:
            fig,ax=plt.subplots()
            
        ax.pcolormesh(self.x,self.y,self.slip)
        
    def MaxSlip(self):
        return np.max(self.slip)
    
    @property
    def is_2d(self):
        return self.x.ndim == 2

#%%
# ──────────────────────────────────────────────────────────────────────────────

class LatLonFaultSegment(FaultSegment):
    """Geographic segment: lat/lon grids in degrees."""
    def __init__(self, *args, surfaceProjection: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.surfaceProjection = surfaceProjection 
        ### #if surface projection  True  I  call UTMFaultSegment_SurfaceProjection which uses rejection algrthim. 
        ### This is mainly for Chile 1960 which is already in surface porjection
        self.UTM_seg=self.to_utm(surfaceProjection)
        
        
    def ReturnHowManyPoints(self):
        return self.UTM_seg.ReturnHowManyPoints()
    
    def Sample(self,N=None):
        points_in_utm=self.UTM_seg.Sample(N) #points in utm
        return self.from_utm(points_in_utm)
    
    def SampleSlip(self,N=None):
        return self.UTM_seg.ReturnSlip(N) 
         
    def from_utm(self,points_in_km):
        
        x_utm = (points_in_km[:, 0] * 1e3).cpu().numpy()
        y_utm = (points_in_km[:, 1] * 1e3).cpu().numpy()
        lon, lat = self.proj(x_utm, y_utm, inverse=True) #convert points to lat/long
        lon = torch.as_tensor(lon, dtype=points_in_km.dtype, device=points_in_km.device)
        lat = torch.as_tensor(lat, dtype=points_in_km.dtype, device=points_in_km.device)
        points_lat_long = torch.stack(( lon,lat), dim=1)
            
        return points_lat_long
    

            
    def to_utm(self,surfaceProjection) -> "UTMFaultSegment":
        """Project this lat/lon grid into UTM metres."""
        φ0 = float(np.nanmean(self.y))
        λ0 = float(np.nanmean(self.x))
        if λ0 > 180:
            λ0 -= 360  # Convert to [-180, 180] if needed
        zone = int((λ0 + 180) / 6) + 1
        proj = Proj(proj="utm", zone=zone, ellps="WGS84")
        epsg = f"EPSG:326{zone}" if φ0 >= 0 else f"EPSG:327{zone}"

        x_m, y_m = proj(self.x.ravel(), self.y.ravel())
        X = x_m.reshape(self.x.shape)/1e3 #convert to km
        Y = y_m.reshape(self.y.shape)/1e3 #convert to km
        self.proj=proj
        return self.CreateUTMProjectionSeg(X,Y,epsg,surfaceProjection)
    
    def CreateUTMProjectionSeg(self,X,Y,epsg,surfaceProjection):
        cls = UTMFaultSegment_SurfaceProjection if surfaceProjection else UTMFaultSegment #if surface projection choose UTM fault seg with surface projection 
        
        return cls(
            slip=self.slip,
            x=X,
            y=Y,
            z=self.z,
            dip=self.dip,
            rake=self.rake,
            strike=self.strike,
            DimWL=self.DimWL,
            position=self.position,
            epsg=epsg,
            additionalData=self.additionalData
        )
    
    
    def PlotContours(self,n_levels=3,ax=None,contour_kwargs=None,clabel_kwargs=None,contours=None):
        if ax is None:
            fig,ax=plt.subplots()
    
        if contour_kwargs is None:
            contour_kwargs = {}
            
            
        contour_set = ax.contour(self.slip, levels=n_levels, extent=self.extent, **contour_kwargs)
        if clabel_kwargs is not None:
            ax.clabel(contour_set, **clabel_kwargs)
            
            
    def Subfaults_collection(self,values="slip", **polygon_args):
        # Get UTM-based polygons and associated slip
        subfault_utm_collection = self.UTM_seg.Subfaults_collection(values=values,**polygon_args)
        arr = subfault_utm_collection.get_array()
        arr.set_fill_value(np.nan)
        arr = arr.filled()
    
        # Extract polygons and convert to lon/lat
        polygons = []
        for path in subfault_utm_collection.get_paths():
            xy = path.vertices
            lon, lat = self.proj(xy[:, 0] * 1e3, xy[:, 1] * 1e3, inverse=True)
            lonlat = np.column_stack([lon, lat])
            polygons.append(mpatches.Polygon(lonlat, closed=True))
    
        # Create a new lon/lat-based PatchCollection
        collection = PatchCollection(polygons, **polygon_args)
        collection.set_array(arr)

        return collection
    
    def TotalSlipArea(self):
        return self.UTM_seg.TotalSlipArea()
    
    def Area(self,minSlip=None):
        return self.UTM_seg.Area(minSlip)
    
    def SampleTime(self,N=None):
        return self.UTM_seg.SampleTime(N)
    
    def ShiftSegment(self,azimuth,distance,geoid):
        lon_pts = self.x    # e.g. shape (nx, nz)
        lat_pts = self.y
        az_arr   = np.full_like(lon_pts,   azimuth, dtype=float)
        dist_arr = np.full_like(lon_pts,   distance, dtype=float)
        lon_new, lat_new, _ = geoid.fwd(lon_pts, lat_pts, az_arr, dist_arr)
        return LatLonFaultSegment(slip=self.slip,x=lon_new,y=lat_new,z=self.z,
                                  dip=self.dip,rake=self.rake,strike=self.strike,
                                  DimWL=self.DimWL,position=self.position,
                                  surfaceProjection=self.surfaceProjection,additionalData=self.additionalData)
    

    
    
    def RotateAroundPoint(self, angle_deg, pivot_lon, pivot_lat, *, rotate_strike_with_body=True):
        from pyproj import CRS, Transformer
        aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={pivot_lat} +lon_0={pivot_lon} +ellps=WGS84")
        wgs84 = CRS.from_epsg(4326)
        fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True)
        inv = Transformer.from_crs(aeqd, wgs84, always_xy=True)

        X, Y = fwd.transform(self.x, self.y)               # metres
        X0, Y0 = fwd.transform(pivot_lon, pivot_lat)

        c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        xr = (X - X0) * c - (Y - Y0) * s + X0
        yr = (X - X0) * s + (Y - Y0) * c + Y0

        lon_new, lat_new = inv.transform(xr, yr)

        strike_new = (self.strike - angle_deg) % 360 if rotate_strike_with_body else self.strike

        return LatLonFaultSegment(
            slip=self.slip, x=lon_new, y=lat_new, z=self.z,
            dip=self.dip, rake=self.rake, strike=strike_new,
            DimWL=self.DimWL, position=self.position,
            surfaceProjection=self.surfaceProjection, additionalData=self.additionalData
        )
        

            
    def ChangeSlipPerArea(self,new_slipPerArea):
        self.UTM_seg.ChangeSlipPerArea(new_slipPerArea)
        
    def HowManySubfaults(self):
        return self.UTM_seg.HowManySubfaults()
    
    def copy(self, slip=None):
        """
        Return a new instance of the same concrete class.
        Only supports overriding `slip` (must match self.slip shape).
        """
        if slip is None:
            slip_new = self.slip.copy()
        else:
            slip_new = np.asarray(slip)
            if slip_new.shape != self.slip.shape:
                raise ValueError(f"slip shape {slip_new.shape} != original {self.slip.shape}")
            slip_new = slip_new.copy()

        # common kwargs (work for all subclasses)
        kwargs = dict(
            slip=slip_new,
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            dip=self.dip,
            rake=self.rake,
            strike=self.strike,
            DimWL=self.DimWL.copy(),
            position=self.position,
            additionalData=self.additionalData.copy(deep=True) if hasattr(self.additionalData, "copy") else self.additionalData,
        )

        # subclass-specific init args
        if hasattr(self, "surfaceProjection"):   # LatLonFaultSegment
            kwargs["surfaceProjection"] = self.surfaceProjection
        if hasattr(self, "epsg"):                # UTMFaultSegment
            kwargs["epsg"] = self.epsg

        return type(self)(**kwargs)
    
    def ClosestIndex(self, lon, lat):
        """
        Convert lon/lat to UTM and return the closest grid index
        from the underlying UTMFaultSegment.
    
        Parameters
        ----------
        lon, lat : float
            Geographic coordinates in degrees.
    
        Returns
        -------
        idx : tuple[int, int]
            Index into self.x / self.y as (i, j).
        dist : float
            Euclidean distance in km in UTM space.
        """
        x_m, y_m = self.proj(lon, lat)
        x_km = x_m / 1e3
        y_km = y_m / 1e3
        return self.UTM_seg.ClosestIndex(x_km, y_km)
            
        

#%%
# ──────────────────────────────────────────────────────────────────────────────
class UTMFaultSegment(FaultSegment):
    """Projected segment: x/y grids in metres."""
    def __init__(self, *args, epsg: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsg = epsg
        self.FaultPatchGenerate()
        
    def FaultPatchGenerate(self):
        #subfaultType=subfaultPatch_area
        subfaultType=subfaultPatch
        self.faultPatches=[]
        area=self.DimWL[0]*self.DimWL[1]
        for i, (slip_i, x_i, y_i, z_i) in enumerate(zip(self.slip.ravel(), self.x.ravel(), self.y.ravel(), self.z.ravel())):
            edges = subfault_corners(x_i, y_i, z_i, self.strike, self.dip, self.DimWL[1], self.DimWL[0], ref_loc=self.position)
            if self.additionalData is not None and not self.additionalData.empty:
                self.faultPatches.append(subfaultType(edges, area, slip_i, additionalData=self.additionalData.loc[i]))
            else:
                self.faultPatches.append(subfaultType(edges, area, slip_i))  
        
    def Sample(self,N=None):
        points=[]
        for fault_patch_i in self.faultPatches:
            points.append(fault_patch_i.Sample(N))
            
        return torch.cat(points, dim=0)
    
    
    def SampleTime(self,N=None):
        time=[]
        for fault_patch_i in self.faultPatches:
            time.append(fault_patch_i.SampleTime(N))
            
        return np.concatenate(time)
    
    def HowManySubfaults(self):
        return len(self.faultPatches)
    
    
    def ReturnHowManyPoints(self):
        N=0
        for fault_patch_i in self.faultPatches:
            N+=fault_patch_i.ReturnHowManyPoints()
        
        return N
    
    def TotalSlipArea(self):
        SA=0
        for fault_patch_i in self.faultPatches:
            SA+=fault_patch_i.SlipArea()
            
        return SA
        
    
    def Subfaults_collection(self, three_D=False, values="slip", **polygon_args):
        
        all_edges = [sf.ReturnCloseEdges() for sf in self.faultPatches]

        if three_D:
            # 3D: feed lists of [x,y,z] into Poly3DCollection
            verts3d = [e.tolist() for e in all_edges]  # each e is (N×3)
            coll = Poly3DCollection(verts3d, **polygon_args)
        else:
            # 2D: drop Z, wrap into mpatches.Polygon
            polys2d = [mpatches.Polygon(e[:, :2], closed=True)
                       for e in all_edges]
            coll = PatchCollection(polys2d, **polygon_args)

        
        if values == "slip":
            arr = np.asarray(self.slip).ravel()
        elif values == "time":
            arr = np.asarray(self.ReturnAverageTimeArray()).ravel()
        else:
            raise ValueError(f"values must be 'slip' or 'time', got {values!r}")
    
        coll.set_array(arr)
        
        return coll
    
    
    
    def Area(self,minSlip=None):
        
        A=0
        for fault_patch_i in self.faultPatches:
            if minSlip is None or fault_patch_i.slip > minSlip:
                A+=fault_patch_i.Area()
        
        return A
    
    
    def ChangeSlipPerArea(self,new_slipPerArea):
        for fault_patch_i in self.faultPatches:
            fault_patch_i.ChangeSlipPerArea(new_slipPerArea)
            
    def ReturnSlip(self,N=None):
        points_slip=[]
        for fault_patch_i in self.faultPatches:
            points_slip.append(fault_patch_i.ReturnSlip(N))
            
        return np.hstack(points_slip)
    
    def ReturnAverageTimeArray(self):
        """
        Return an array with AverageTime() for all fault patches.
    
        Raises
        ------
        ValueError
            If any patch returns NaN.
        """
        time = np.array([fp.AverageTime() for fp in self.faultPatches], dtype=float)
    
        if np.isnan(time).any():
            raise ValueError("found np.nan in AverageTime() for one or more fault patches")
    
        return time

    
    def ClosestIndex(self, x, y):
        """
        Return the closest grid index in (row, col) to the input UTM x/y,
        together with the Euclidean distance in km.
    
        Parameters
        ----------
        x, y : float
            UTM coordinates in km.
    
        Returns
        -------
        idx : tuple[int, int]
            Index into self.x / self.y as (i, j).
        dist : float
            Euclidean distance in km.
        """
        dx = self.x - x
        dy = self.y - y
        dist2 = dx**2 + dy**2
    
        flat_idx = np.nanargmin(dist2)
        idx = np.unravel_index(flat_idx, self.x.shape)
        dist = np.sqrt(dist2[idx])
    
        return idx, dist
        
            

    
    
    
#%%
class UTMFaultSegment_SurfaceProjection(UTMFaultSegment):
    def FaultPatchGenerate(self):
        if self.is_2d:
            self.sampler=to.gridBasedDistribution(X=self.x,Y=self.y,values=self.slip)
        else:
            self.sampler=to.generateGridBaseDistributionFromScatter(self.x.ravel(),self.y.ravel(), self.slip.ravel(),grid_size=[100,100],method='linear')  ### this is a rejection algrthim
    def Sample(self,N=None):
        if N is None:
            N=self.ReturnHowManyPoints()
        return self.sampler.Sample(N,normalize=False)
    
    def ReturnHowManyPoints(self):
        slipPerArea = getattr(self, "slipPerArea", SLIP_PER_AREA)
        dx=self.sampler.interpolator.X[0,1]-self.sampler.interpolator.X[0,0]
        dy=self.sampler.interpolator.Y[0,0]-self.sampler.interpolator.Y[1,0]
        slip=self.sampler.interpolator.values
        areaslip = torch.abs(dx * dy) * slip.sum()
        return int(slipPerArea*areaslip)
        
            
          
#%%
class subfaultPatch:
    def __init__(self,edges,area,slip,slipPerArea=None,additionalData=pd.DataFrame({})):
        self.edges=edges
        self.area=area
        self.slip=slip
        self.sampler=to.FastUniformPolySampler(self.edges[:,0:2])
        self.slipPerArea = slipPerArea if slipPerArea is not None else SLIP_PER_AREA
        self.additionalData=additionalData
        
    def Sample(self,N=None):
        if N is None:
            N=self.ReturnHowManyPoints()
        return self.sampler(N)
    
    def ReturnHowManyPoints(self):
        return int(self.SlipArea()*self.slipPerArea)
    
    def ReturnCloseEdges(self):
        closed_edges = np.vstack([self.edges, self.edges[0]])
        return closed_edges
    
    
    def SlipArea(self):
        return self.area*self.slip
    
    def Area(self):
        return self.area
    
    def ChangeSlipPerArea(self,new_slipPerArea):
        self.slipPerArea=new_slipPerArea
        
    def AverageTime(self):
        if self.additionalData.empty:
            return np.nan
        
        elif {"TRUP", "RISE"}.issubset(self.additionalData.index):
            trup = float(self.additionalData["TRUP"])
            rise = float(self.additionalData["RISE"])
            return trup + 0.5 * rise
        else: 
            #print("found some data but not TRUP and RISE",flush=True)
            return np.nan
            
    def SampleTime(self,N=None):
        if N is None:
            N=self.ReturnHowManyPoints()
        
        return np.ones(N)*self.AverageTime()
    
    def ReturnSlip(self,N=None):
        if N is None:
            N=self.ReturnHowManyPoints() 
        return self.slip*np.ones(N)


#%% 
class subfaultPatch_area(subfaultPatch):
    def SlipArea(self):
        return self.area
    
                               
            
            
            
        

#%%
class finiteFault:
    """
    Read ETH/USGS *.mat finite‑fault solutions (Wald & Heaton format) and
    expose metadata + FaultSegment objects.

    Parameters
    ----------
    filename : str
        Path to the *.mat file.
    device, dataType : passed through for any GPU / dtype logic you
        already use elsewhere.
    """
    def __init__(self, metadata,segments):
        self.segments=segments
        self.metadata=metadata
        
    def TotalSlipArea(self):
        SA=0
        for fault_patch_i in self.segments:
            SA+=fault_patch_i.TotalSlipArea()
        
        return SA
    
    def DealWithSamples(self,listOfSamples,convertToNumPy=False):
        total_rows = sum(s.shape[0] for s in listOfSamples)
    
        # 2) Allocate output (on the same device/dtype as first sample—or CPU if no samples)
        if listOfSamples:
            first = listOfSamples[0]
            out = first.new_empty((total_rows, 2))
        else:
            out = torch.empty((0, 2))
    
        # 3) Copy each segment’s sample into its slice of `out`
        offset = 0
        for s in listOfSamples:
            L = s.shape[0]
            out[offset : offset + L, :] = s
            offset += L
    
        # 4) Return as NumPy if requested
        return out.detach().cpu().numpy() if convertToNumPy else out
        
    def Sample(self,N=None,convertToNumPy=False):
        return self.DealWithSamples([seg.Sample(N) for seg in self.segments],convertToNumPy)
    
    
    def SampleSlip(self,N=None):
        return np.hstack([seg.SampleSlip(N) for seg in self.segments])
    
    def SampleTime(self,N=None,normalize=False):
        time = [seg.SampleTime(N) for seg in self.segments]
        time=np.concatenate(time)
        return time/np.max(time) if normalize else time
             
    
    def ShiftEQ(self,new_epicenter_long,new_epicenter_lat):
        geod = Geod(ellps="WGS84")
        az_shift, _, dist_shift = geod.inv(self.metadata['long'], self.metadata['lat'], 
                                           new_epicenter_long, new_epicenter_lat)
        
        new_segments=[]
        for seg_i in self.segments:
            new_segments.append(seg_i.ShiftSegment(az_shift,dist_shift,geod))
            
        shifteed_metadata=self.metadata.copy()
        shifteed_metadata.update({ 'lat':  new_epicenter_lat, 'long': new_epicenter_long})
        
        return finiteFault(shifteed_metadata, new_segments)
    
    
    def RotateAroundPoint(self, angle_deg: float, lon0: float | None = None, lat0: float | None = None, *, rotate_strike_with_body: bool = True) -> "finiteFault":
        "will rotate around lon0 lat0 set to epicenter if none"
        
        lon0 = self.metadata["long"] if lon0 is None else lon0
        lat0 = self.metadata["lat"]  if lat0 is None else lat0

        new_segments = []
        for seg in self.segments:
            rotated_seg = seg.RotateAroundPoint(angle_deg,pivot_lon=lon0,pivot_lat=lat0,rotate_strike_with_body=rotate_strike_with_body)
            new_segments.append(rotated_seg)

        return finiteFault(self.metadata.copy(), new_segments)
            
    def TotalArea(self):
        A=0
        for seg_i in self.segments:
            A+=seg_i.Area()
        return A
    
    def HowManySubfaults(self):
        N=0
        for seg_i in self.segments:
            N+=seg_i.HowManySubfaults()
        return N 

    def ChangeSlipPerArea(self,new_slipPerArea):
        for seg_i in self.segments:
            seg_i.ChangeSlipPerArea(new_slipPerArea)

    def ReturnHowManyPoints(self):
        N=0
        for seg_i in self.segments:
            N+=seg_i.ReturnHowManyPoints()
        return N
    
    def Segments_collection(self, values="slip", **polygon_args):
        if "norm" not in polygon_args:
            if values == "slip":
                vmax = max(np.nanmax(np.asarray(seg_i.slip)) for seg_i in self.segments)
                polygon_args["norm"] = Normalize(vmin=0, vmax=vmax)
            elif values == "time":
                vmax = max(np.nanmax(seg_i.UTM_seg.ReturnAverageTimeArray()) for seg_i in self.segments)
                polygon_args["norm"] = Normalize(vmin=0, vmax=vmax)
            else:
                raise ValueError(f"values must be 'slip' or 'time', got {values!r}")
    
        collections = []
        for seg_i in self.segments:
            collections.append(seg_i.Subfaults_collection(values=values, **polygon_args))
        return collections
    
    def in_poly(self,poly, pt, eps=1e-5):
        return (poly.buffer(eps).covers(pt)) if eps else poly.covers(pt)
    
    def FindEpiCenterSubfault(self,lon=None, lat=None):
        from shapely.geometry import Point, Polygon as ShapelyPolygon
        collections=self.Segments_collection()
        
        if lon is None:
            lon = self.metadata["long"]
        if lat is None:
            lat = self.metadata["lat"]
            
        epi_center = Point(lon, lat)

        subfaults=[]
        segments=[]
        for j,coll_i in enumerate(collections):
            for i, patch in enumerate(coll_i.get_paths()):
                verts = patch.vertices  # shape (N, 2)
                shapely_poly = ShapelyPolygon(verts)
                if self.in_poly(shapely_poly,epi_center):#shapely_poly.contains(epi_center):
                    subfaults.append(self.segments[j].UTM_seg.faultPatches[i])
                    segments.append(self.segments[j])                    
        if len(subfaults)==0:
            raise ValueError("can't find epicenter in subfaults")
            
        return segments,subfaults
    
    def SampleUniform(self,N=1,convertToNumPy=True):
        numOfSamplesSubfault=int(np.ceil(N/self.HowManySubfaults()))
        unifrom_samples=self.DealWithSamples([seg.Sample(numOfSamplesSubfault) for seg in self.segments],convertToNumPy)
        
        if len(unifrom_samples)<N:
            raise ValueError("didn't get enough samples")
            
        np.random.default_rng().shuffle(unifrom_samples, axis=0)
        return unifrom_samples[0:N]
        
    
    def Sample_Epicenter(self,N=None,convertToNumPy=False):
        segments,subfaults=self.FindEpiCenterSubfault()
        samples=[]
        for segments_j,subfault_j in zip(segments,subfaults):
            if N==None:
                N=int(subfault_j.slipPerArea*subfault_j.area)
            samples.append(segments_j.from_utm(subfault_j.Sample(N)))
                    
        return self.DealWithSamples(samples,convertToNumPy)
        
    
    def PlotSubFaultEpicenter(self,ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
            
        all_lons  = np.concatenate([
            np.atleast_1d(seg.coords_x).ravel() for seg in self.segments
        ])
        all_lats  = np.concatenate([
            np.atleast_1d(seg.coords_y).ravel() for seg in self.segments
        ])
        
        
        ax.scatter(all_lons,all_lats,**kwargs)
    
        
        
            
    def PlotEpicenter(self, ax=None, **kwargs):
        lat = self.metadata.get("lat")
        lon = self.metadata.get("long")
        evTAG =self.metadata.get("EventTAG")
        date = self.metadata.get("Date", "")
        auth = self.metadata.get("Reference", "")
        if lat is None or lon is None:
            return
    
        if ax is None:
            _, ax = plt.subplots()
    
        ax.scatter(lon, lat, marker="+", color="magenta", s=35,**kwargs)
        ax.text(lon, lat, f"{date}\n{auth}\n{evTAG}", ha="left", color='magenta',va="bottom", **kwargs)
        return ax
    
    def IntepolateSegemnts(self,grid_res=200,interp_method='linear'):
        # 1) Collect all points
        all_lons  = np.concatenate([
            np.atleast_1d(seg.coords_x).ravel() for seg in self.segments
        ])
        all_lats  = np.concatenate([
            np.atleast_1d(seg.coords_y).ravel() for seg in self.segments
        ])
        all_slips = np.concatenate([
            np.atleast_1d(seg.slip).ravel()     for seg in self.segments
        ])

        # 2) Build uniform grid
        lon_min, lon_max = all_lons.min(), all_lons.max()
        lat_min, lat_max = all_lats.min(), all_lats.max()
        xi = np.linspace(lon_min, lon_max, grid_res)
        yi = np.linspace(lat_min, lat_max, grid_res)
        X, Y = np.meshgrid(xi, yi)

        # 3) Interpolate slip onto grid
        Z = griddata(
            points=(all_lons, all_lats),
            values=all_slips,
            xi=(X, Y),
            method=interp_method,
            fill_value=np.nan
        )
        
        return X,Y,Z
    
    
    def PlotContoursKDE(self,
                     n_levels=2,
                     ax=None,
                     contour_kwargs=None):
        if ax is None:
            fig, ax = plt.subplots()
            
        samples=self.Sample(convertToNumPy=True)
        local_kwargs = {} if contour_kwargs is None else contour_kwargs.copy()
        
        
        if n_levels == None:
            local_kwargs.update({'levels': [0.2], 'thresh': 0})
        else:
            local_kwargs.update({'levels':n_levels})
            
        local_kwargs.update({'ax':ax})
        
        sns.kdeplot(x=samples[:,0],y=samples[:,1],**local_kwargs)
            
        
    

    def PlotContours(self,
                     #segment_index=None,
                     n_levels=2,
                     ax=None,
                     contour_kwargs=None,
                     clabel_kwargs=None,
                     #contours=None,
                     grid_res=200,
                     interp_method='linear',level_to_show=0.3):
        # Delegate single segment
        # if segment_index is not None:
        #     return self.segments[segment_index].PlotContours(
        #         n_levels=n_levels, ax=ax,
        #         contour_kwargs=contour_kwargs,
        #         clabel_kwargs=clabel_kwargs,
        #         contours=contours
        #     )

        if ax is None:
            fig, ax = plt.subplots()

        # color logic
        local_kwargs = {} if contour_kwargs is None else contour_kwargs.copy()
    
        # default color logic on the local copy
        #if 'color' not in local_kwargs:
        #    auth = self.metadata.get("invAUTH", "")
        #    local_kwargs['color'] = 'cyan' if "HAYES" in auth.upper() else 'green'
    
        # translate 'color' → 'colors' for Cartopy/Matplotlib
        #if 'color' in local_kwargs:
        #    local_kwargs['colors'] = local_kwargs.pop('color')

        X,Y,Z=self.IntepolateSegemnts(grid_res=grid_res,interp_method=interp_method)
        
        

# 2) pad Z by one cell on every side
# 1) pick your pad value (here zero)
        pad_val = 0.
        
        # 2) create a padded Z
        Z_filled = np.nan_to_num(Z, nan=pad_val)
        Z2 = np.pad(Z_filled,
                    pad_width=((1,1),(1,1)),         # one row/col on each side
                    mode='constant',
                    constant_values=pad_val)
        
        # 3) build matching X2, Y2 by extending your original grid spacing
        dx = X[0,1] - X[0,0]
        dy = Y[1,0] - Y[0,0]
        
        x2 = np.concatenate([[X[0,0]-dx], X[0,:], [X[0,-1]+dx]])
        y2 = np.concatenate([[Y[0,0]-dy], Y[:,0], [Y[-1,0]+dy]])
        X2, Y2 = np.meshgrid(x2, y2)
        
        # 4) contour the padded grid
        


        # 4) Compute contour levels
        #levels = contours if contours is not None else ComputeLevels(Z, n_levels=n_levels)

        # 5) Plot
        if n_levels == None:
            levels=[np.nanmax(Z)*level_to_show]
        else:
            levels=n_levels
        cs = ax.contour(X2, Y2, Z2, levels=levels, **local_kwargs)
        

#%%

    

#%%



#%% read srf EQs


# Example usage:
# data = read_srf_group_by_geometry("Yokota_2011.srf")
# print(data['groups'].keys())  # e.g., {(strike1, dip1), (strike2, dip2), ...}
# For group in data['groups'], you can inspect 'header', 'time_params', 'slip_rates' for that strike/dip.

# Example usage:
# data = read_srf_auto_grid("Yokota_2011.srf")
# If the header contains "16 x 6", data['grid_shape'] == (6, 16)
# Then data['header']['lon'] will have shape (6, 16) and slip_rates (6, 16, max_samples).
