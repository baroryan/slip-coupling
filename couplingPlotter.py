import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LightSource
from matplotlib import rcParams
import geopandas as gpd
import os
import torchOperators as to
import json
from typing import List, Tuple, Optional
from matplotlib.path import Path
from cartopy.mpl.geoaxes import GeoAxes
#import closet_distance_to_polygon
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.ticker import FixedLocator
import matplotlib.ticker as mticker
import warnings
#%%
def getFullPath(relativePath):
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(MODULE_DIR, relativePath)

#%%
def ConvertSamplerToLocalSegment(torchSampler):
    return CouplingObjectGrid(lon=torchSampler.interpolator.X.numpy(),lat=torchSampler.interpolator.Y.numpy(),coupling=torchSampler.interpolator.values.numpy())
#%%
#--------------------------------------------------
# 1. Compute Boundaries (using get_extent() if available)
#--------------------------------------------------
def compute_boundaries(coupling_obj_list, lon_min=None, lon_max=None, lat_min=None, lat_max=None,margin_lon_ratio=0.02,margin_lat_ratio=0.02):
    """
    Computes boundaries from coupling objects that implement get_extent().
    Only replaces values that are None; retains provided values.
    Adds a 2% margin to computed parts.
    """
    extents = [obj.get_extent() for obj in coupling_obj_list if hasattr(obj, 'get_extent')]
    if not extents:
        raise ValueError("No valid objects with get_extent() found.")

    # Combine all object extents
    all_lons = [e[0] for e in extents] + [e[1] for e in extents]
    all_lats = [e[2] for e in extents] + [e[3] for e in extents]

    # Compute global min/max
    global_lon_min, global_lon_max = np.min(all_lons), np.max(all_lons)
    global_lat_min, global_lat_max = np.min(all_lats), np.max(all_lats)

    # Compute margins
    margin_lon = margin_lon_ratio * (global_lon_max - global_lon_min)
    margin_lat = margin_lat_ratio * (global_lat_max - global_lat_min)

    # Fill in missing values only
    final_lon_min = lon_min if lon_min is not None else global_lon_min - margin_lon
    final_lon_max = lon_max if lon_max is not None else global_lon_max + margin_lon
    final_lat_min = lat_min if lat_min is not None else global_lat_min - margin_lat
    final_lat_max = lat_max if lat_max is not None else global_lat_max + margin_lat

    return final_lon_min, final_lon_max, final_lat_min, final_lat_max


#def return_lon_right(lon):
#        return lon if lon <= 180 else lon - 360
    
def return_lon_right(lon):
    lon = np.asarray(lon)
    return np.where(lon <= 180, lon, lon - 360)

#%%
#%%
#--------------------------------------------------
# 2. CouplingObjectGrid with get_extent() and pcolormesh
#--------------------------------------------------
class CouplingObjectGrid:
    def __init__(self, lat, lon, coupling):
        """
        Create a CouplingObjectGrid with lat, lon, and coupling data.
        If lat/lon are 1D, they are converted to 2D via meshgrid.
        If lat/lon are 2D, they are used as is.
        """
        if lat.ndim == 1 and lon.ndim == 1:
            if coupling.shape != (lat.size, lon.size):
                raise ValueError("For 1D lat/lon, 'coupling' must have shape (len(lat), len(lon)).")
                
            self.data_array = xr.DataArray(
                coupling, dims=["lat", "lon"],
                coords={"lat": lat, "lon": lon})
        
        elif lat.ndim == 2 and lon.ndim == 2:
            if coupling.shape != lat.shape:
                raise ValueError("For 2D lat/lon, 'coupling' shape must match lat/lon shapes.")
            # For 2D, assume they are rectilinear and extract 1D vectors.
            if not np.allclose(lat, np.repeat(lat[:, [0]], lat.shape[1], axis=1)):
                raise ValueError("2D lat array is not uniform along columns.")
            if not np.allclose(lon, np.repeat(lon[[0], :], lon.shape[0], axis=0)):
                raise ValueError("2D lon array is not uniform along rows.")
            lat_vec = lat[:, 0]
            lon_vec = lon[0, :]
            if coupling.shape != (lat_vec.size, lon_vec.size):
                raise ValueError("Coupling shape mismatch with extracted vectors.")
            self.data_array = xr.DataArray(
                coupling, dims=["lat", "lon"],
                coords={"lat": lat_vec, "lon": lon_vec}
            )
        else:
            raise ValueError("lat and lon must be either both 1D or both 2D arrays.")
            
        lat_vals = self.data_array["lat"].values
        dlat = np.diff(lat_vals)
        if np.all(dlat > 0):
            pass  # already increasing
        elif np.all(dlat < 0):
            self.data_array = self.data_array.sortby("lat")  # flips both coord and data
        else:
            raise ValueError("lat must be strictly monotonic")
        
        self.mask_nan_flat=np.isnan(coupling.ravel())

    def get_extent(self):
        """
        Returns the (lon_min, lon_max, lat_min, lat_max) of the object's coordinates.
        """
        lat_vals = self.data_array.coords["lat"].values
        lon_vals = self.data_array.coords["lon"].values
        
        
        
        lon_min=float(np.nanmin(lon_vals));query_lon_min = return_lon_right(lon_min)
        lon_max=float(np.nanmax(lon_vals));query_lon_max = return_lon_right(lon_max)
        
        if query_lon_min > query_lon_max:
            query_lon_min, query_lon_max = query_lon_max, query_lon_min
        
        return (query_lon_min, query_lon_max,
                float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals)))

    def plot_segment(self, ax=None, style_params=None):
        """
        Plot the coupling data using pcolormesh. pcolormesh handles dateline-crossing naturally.
        """
        if style_params is False:
            style_params={}
        elif style_params is None:
            style_params = {
                "cmap": "hot_r",
                "vmin": 0,
                "vmax": 1,
                "alpha": 1
            }
        created_new = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6),
                                   subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            created_new = True

        # Prepare 2D coordinate arrays.
        lon_vals = self.data_array.coords["lon"].values
        lat_vals = self.data_array.coords["lat"].values
        if lon_vals.ndim == 1 and lat_vals.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        else:
            lon2d, lat2d = lon_vals, lat_vals
        cmap=style_params.get("cmap", "hot_r")
        

        pm = ax.pcolormesh(
            lon2d, lat2d, self.data_array.values,
            transform=ccrs.PlateCarree(),**style_params)
        
        if created_new:
            return fig, ax
        return ax

    def __repr__(self):
        da = self.data_array
        return (f"CouplingObjectGrid(\n"
                f"  lat shape: {da.coords['lat'].values.shape},\n"
                f"  lon shape: {da.coords['lon'].values.shape},\n"
                f"  coupling shape: {da.values.shape}\n)")
    
    
    def plot_contour(self, ax=None, contour_params=None):
        """
        Plot the coupling data as contour lines using ax.contour.
        
        Parameters:
            ax : matplotlib.axes.Axes, optional
                The axis on which to plot. If None, a new axis is created.
            contour_params : dict, optional
                A dictionary of contour plotting parameters. Defaults are:
                    levels: 20
                    colors: 'black'
                    linewidths: 1
                    linestyles: 'solid'
                    alpha: 1
        Returns:
            If a new axis is created, returns (fig, ax); otherwise, returns ax.
        """
        default_params = {
            "levels": 10,
            "colors": "magenta",
            "linewidths": 1,
            "linestyles": "--",
            "alpha": 1
        }
        if contour_params is None:
            contour_params = {}
        params = {**default_params, **contour_params}
        
        created_new = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6),
                                   subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            created_new = True
    
        # Prepare 2D coordinate arrays.
        lon_vals = self.data_array.coords["lon"].values
        lat_vals = self.data_array.coords["lat"].values
        if lon_vals.ndim == 1 and lat_vals.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        else:
            lon2d, lat2d = lon_vals, lat_vals
    
        cs = ax.contour(
            lon2d, lat2d, self.data_array.values,
            transform=ccrs.PlateCarree(),
            **params
        )
        
        if created_new:
            return fig, ax
        return ax
    
    def plot_scatter_points(self, x, y, ax=None, scatter_params=None):
        """
        Plot scatter points on the map using given x and y coordinates.
        
        Parameters:
            x : array-like
                Longitudes of the points.
            y : array-like
                Latitudes of the points.
            ax : matplotlib.axes.Axes, optional
                The axis on which to plot. If None, a new axis with 
                PlateCarree(central_longitude=180) projection is created.
            scatter_params : dict, optional
                Dictionary of scatter plot parameters. Defaults:
                {
                  "marker": "o",
                  "s": 10,
                  "color": "blue",
                  "edgecolors": "none"
                }
        
        Returns:
            If a new axis is created, returns (fig, ax); otherwise, returns ax.
        """
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        
        # Set default scatter parameters
        if scatter_params is None:
            scatter_params = {"marker": "o", "s": 10, "color": "blue", "edgecolors": "none"}
        
        created_new = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6),
                                   subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            created_new = True
    
        sc = ax.scatter(x, y, transform=ccrs.PlateCarree(), **scatter_params)
        
        if created_new:
            return fig, ax
        return ax
    
    def ReturnCouplingSampler(self,device='cpu'):
        lon2d, lat2d=self.GetLon2dLat2d()
        return to.gridBasedDistribution(lon2d, lat2d, self.data_array.values,device=device)
    
    
    def LoadSegments(self,segmentManger):
        self.segmentManger=segmentManger
        
    def GetLon2dLat2d(self):
        lon_vals = self.data_array.coords["lon"].values
        lat_vals = self.data_array.coords["lat"].values
        if lon_vals.ndim == 1 and lat_vals.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        else:
            lon2d, lat2d = lon_vals, lat_vals
            
        return lon2d, lat2d
    
    def GenerateObjectWithDifferentData(self,data):
        return CouplingObjectGrid(lat=self.data_array.lat,lon=self.data_array.lon,coupling=data)
        
            
    def ReturnFlatData(self):
        return self.data_array.values.ravel()
    
    
    def InterpolateToLocalLong(self, xarrayToInterpolateTo, *, max_radius_deg=30, step_deg=5, verbose=True):
        """
        Interpolate *xarrayToInterpolateTo* onto the non-NaN nodes of self.data_array (lat×lon).
        Pipeline:
          1) linear interpolation
          2) nearest-neighbor fallback
          3) KDTree nearest fill with expanding search radius (step_deg -> max_radius_deg)
    
        Returns a NEW CouplingObjectGrid with values placed only at points where self.data_array is finite.
        """
        import numpy as np
        import warnings
        import xarray as xr
        from scipy.spatial import cKDTree
    
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
    
        # --- 0) Sanity & prep ---------------------------------------------------
        if not {"lat", "lon"}.issubset(xarrayToInterpolateTo.coords):
            raise ValueError("xarrayToInterpolateTo must have lat/lon coordinates")
    
        src = xarrayToInterpolateTo
        if isinstance(src, xr.Dataset):
            # Prefer a sensible default name if present
            if "depth" in src.data_vars:
                src = src["depth"]
            else:
                # take the first variable
                name0 = next(iter(src.data_vars))
                src = src[name0]
    
        # ensure numeric dtype
        src = src.astype("float64")
    
        # --- 1) Bounding box overlap -------------------------------------------
        lat_self_min, lat_self_max = float(self.data_array.lat.min()), float(self.data_array.lat.max())
        lon_self_min, lon_self_max = float(self.data_array.lon.min()), float(self.data_array.lon.max())
        lat_src_min,  lat_src_max  = float(src.lat.min()) , float(src.lat.max())
        lon_src_min,  lon_src_max  = float(src.lon.min()) , float(src.lon.max())
    
        overlap = (
            (lat_self_max >= lat_src_min)  and (lat_self_min <= lat_src_max) and
            (lon_self_max >= lon_src_min)  and (lon_self_min <= lon_src_max)
        )
        if not overlap:
            raise ValueError("No coordinate overlap between the two grids.")
    
        # --- 2) Target points (finite in self) ----------------------------------
        base_vals = np.asarray(self.data_array.values)
        valid_mask = np.isfinite(base_vals)
        if not valid_mask.any():
            raise ValueError("self.data_array contains only NaNs — nothing to interpolate.")
    
        lat_grid, lon_grid = np.meshgrid(self.data_array.lat.values,
                                         self.data_array.lon.values,
                                         indexing="ij")
        targ_lat = lat_grid[valid_mask]
        targ_lon = lon_grid[valid_mask]
    
        vprint(f"[InterpolateToLocalLong] Valid target points: {targ_lat.size}")
    
        # Convenience “points” coords
        pts_lat = xr.DataArray(targ_lat, dims="points")
        pts_lon = xr.DataArray(targ_lon, dims="points")
    
        # --- 3) Linear interpolation --------------------------------------------
        vprint("[InterpolateToLocalLong] Doing linear interpolation…")
        lin_da = src.interp(lat=pts_lat, lon=pts_lon, method="linear")
        nans_after_linear = int(lin_da.isnull().sum().item())
        vprint(f"[InterpolateToLocalLong] NaNs after linear: {nans_after_linear}")
    
        filled = lin_da
    
        # --- 4) Nearest-neighbor fallback --------------------------------------
        if nans_after_linear > 0:
            vprint("[InterpolateToLocalLong] Found NaNs in linear; doing nearest-neighbor fallback…")
            nn_da = src.interp(lat=pts_lat, lon=pts_lon, method="nearest")
            filled = lin_da.where(~lin_da.isnull(), nn_da)
            nans_after_nearest = int(filled.isnull().sum().item())
            vprint(f"[InterpolateToLocalLong] NaNs after nearest: {nans_after_nearest}")
        else:
            nans_after_nearest = 0
    
        # --- 5) KDTree nearest with expanding radius ----------------------------
        still_need = filled.isnull()
        if still_need.any().item():
            vprint("[InterpolateToLocalLong] Still have NaNs; doing KDTree nearest with expanding radius…")
    
            # Build KDTree on finite source nodes
            src_vals = np.asarray(src.values)
            s_lat2d, s_lon2d = np.meshgrid(src.lat.values, src.lon.values, indexing="ij")
            src_mask = np.isfinite(src_vals)
    
            if not src_mask.any():
                warnings.warn("Source contains only NaNs; cannot fill.", RuntimeWarning)
            else:
                s_lat = s_lat2d[src_mask]
                s_lon = s_lon2d[src_mask]
                s_val = src_vals[src_mask]
    
                # scale lon by cos(lat) so degrees behave ~isotropic
                def _xy(lon, lat):
                    cl = np.cos(np.deg2rad(lat))
                    return np.column_stack((lon * cl, lat))
    
                tree = cKDTree(_xy(s_lon, s_lat))
    
                # unresolved target points
                t_lat = targ_lat[still_need.values]
                t_lon = targ_lon[still_need.values]
                t_xy  = _xy(t_lon, t_lat)
    
                filled_vals = np.asarray(filled.values).copy()
                unresolved = np.where(still_need.values)[0]     # indices into the 1D "points" vector
                resolved = np.zeros(unresolved.size, dtype=bool)
    
                for R in range(step_deg, max_radius_deg + step_deg, step_deg):
                    sel_unres = np.flatnonzero(~resolved)
                    if sel_unres.size == 0:
                        break
    
                    dist, idx = tree.query(t_xy[sel_unres], k=1, workers=-1)
                    pick = dist <= R
                    if pick.any():
                        chosen = sel_unres[pick]               # positions within unresolved
                        point_idx = unresolved[chosen]         # 1D indices in the "points" array
                        filled_vals[point_idx] = s_val[np.asarray(idx[pick]).ravel()]
                        resolved[pick] = True
                        vprint(f"[InterpolateToLocalLong] KDTree: filled {pick.sum()} points at radius ≤ {R}°")
    
                remaining = int((~resolved).sum())
                vprint(f"[InterpolateToLocalLong] NaNs remaining after KDTree: {remaining}")
                if remaining > 0:
                    warnings.warn(
                        "Interpolation produced NaNs even after KDTree fallback with expanded search.",
                        RuntimeWarning
                    )
    
                filled = xr.DataArray(filled_vals, dims=["points"])
    
        # --- 6) Reinsert into full 2D grid -------------------------------------
        new_vals = np.full(self.data_array.shape, np.nan, dtype="float64")
        new_vals[valid_mask] = np.asarray(filled.values, dtype="float64")
    
        vprint("[InterpolateToLocalLong] Done. Constructing new grid…")
    
        # --- 7) Return new CouplingObjectGrid ----------------------------------
        return self.__class__(self.data_array.lat.values,
                              self.data_array.lon.values,
                              new_vals)
    
    def downsample(self, N):
        lat, lon = self.data_array.lat.values[::N], self.data_array.lon.values[::N]
        return CouplingObjectGrid(lat, lon, self.data_array.values[::N, ::N])
    
    

    def compare_shapes(self, objects):
        """Ensure each object's data_array shape and mask_nan_flat match self's."""
        
    
        ref_shape = self.data_array.values.shape
        ref_mask  = np.asarray(self.mask_nan_flat, bool).ravel()
        if ref_mask.size != self.data_array.values.size:
            raise ValueError("self.mask_nan_flat length must equal self.data_array.size")
    
        for i, obj in enumerate(objects):
            if obj.data_array.values.shape != ref_shape:
                raise ValueError(f"objects[{i}] shape {obj.data_array.values.shape} != {ref_shape}")
            m = np.asarray(obj.mask_nan_flat, bool).ravel()
            if m.size != ref_mask.size:
                raise ValueError(f"objects[{i}] mask length {m.size} != {ref_mask.size}")
            diff = np.flatnonzero(m != ref_mask)
            if diff.size:
                raise ValueError(f"objects[{i}] mask mismatch at flat index {int(diff[0])}")
        return True

#%%
#--------------------------------------------------
# 3. CouplingMapPlotter using compute_boundaries & pcolormesh
#--------------------------------------------------
class CouplingMapPlotter:
    def __init__(self, coupling_obj_list, lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                 projection=None, style_params=None, load_hillshade=True):
        """
        Initializes a map plotter for coupling objects.
        Global boundaries are computed using each object's get_extent() (with a 2% margin).
        Defaults to PlateCarree(central_longitude=180) projection.
        """
        default_style = {
            "land_color": "beige",
            "ocean_color": "lightblue",
            "land_alpha": 0.3,
            "ocean_alpha": 0.3,
            "hillshade_alpha": 0.1,
            "cmap": "hot_r",
            "coupling_vmin": 0,
            "coupling_vmax": 1,
            "coupling_alpha": 1,
            "xtick_major": 3,
            "ytick_major": 3,
            "xtick_minor": 0.3,
            "ytick_minor": 0.3,
            "margin_ratio":0.02
            
        }
        if style_params is None:
            style_params = {}
        self.style_params = {**default_style, **style_params}
        self.coupling_style= {k: self.style_params[v] for k, v in {"cmap": "cmap", "vmin": "coupling_vmin", "vmax": "coupling_vmax", "alpha": "coupling_alpha"}.items()}

        self.lon_min, self.lon_max, self.lat_min, self.lat_max = compute_boundaries(
            coupling_obj_list, lon_min, lon_max, lat_min, lat_max,self.style_params["margin_ratio"],self.style_params["margin_ratio"]
        )
        self.coupling_obj_list = coupling_obj_list

        if projection is None:
            projection = ccrs.PlateCarree(central_longitude=180)
        self.projection = projection



        self.hillshade = None
        if load_hillshade:  # streams ETOPO1 from NOAA (needs internet)
            self._load_and_create_hillshade()


    
    def _load_and_create_hillshade(self):
        print("Loading ETOPo data. This may take a moment...")
        etopo = xr.open_dataset("https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO1_Ice_g_gmt4.nc")
        
        # Adjust boundaries: force both lon_min and lon_max into the [-180, 180] domain.
        query_lon_min = return_lon_right(self.lon_min)#self.lon_min if self.lon_min <= 180 else self.lon_min - 360
        query_lon_max = return_lon_right(self.lon_max)
        #query_lon_max = self.lon_max if self.lon_max <= 180 else self.lon_max - 360
        # Ensure the minimum is less than the maximum.
        if query_lon_min > query_lon_max:
            query_lon_min, query_lon_max = query_lon_max, query_lon_min
    
        print("Using ETOPo slice boundaries (lon):", query_lon_min, query_lon_max)
        print("Using ETOPo slice boundaries (lat):", self.lat_min, self.lat_max)
        
        # Slice ETOPo using these adjusted boundaries.
        topo = etopo.sel(lon=slice(query_lon_min, query_lon_max),
                         lat=slice(self.lat_min, self.lat_max))
        
        # Check if the subset is empty.
        if topo.z.size == 0:
            raise ValueError("The selected ETOPo subset is empty. Check your boundaries.")
        
        # Save the native ETOPo coordinate arrays.
        self.hillshade_lon = topo.lon.values
        self.hillshade_lat = topo.lat.values
        
        # Compute cell sizes.
        d_lon = np.abs(np.diff(self.hillshade_lon)).mean() if self.hillshade_lon.size > 1 else 0
        d_lat = np.abs(np.diff(self.hillshade_lat)).mean() if self.hillshade_lat.size > 1 else 0
    
        # Extend the extent by half a cell + extra margin to avoid white boxes
        extra = 2.0  # degrees of extra margin
        hillshade_extent = [
            np.nanmin(self.hillshade_lon) - d_lon/2 - extra,
            np.nanmax(self.hillshade_lon) + d_lon/2 + extra,
            np.nanmin(self.hillshade_lat) - d_lat/2 - extra,
            np.nanmax(self.hillshade_lat) + d_lat/2 + extra
        ]
        
        # Compute hillshade.
        elevation = topo.z.values
        ls = LightSource(azdeg=315, altdeg=45)
        self.hillshade = ls.hillshade(elevation, vert_exag=0.005, dx=1, dy=1)
        
        print("Final hillshade extent:", hillshade_extent)
        print("Hillshade successfully created.")
        
        # Store the computed extent for later use in plotting.
        self.hillshade_extent = hillshade_extent
        
        
    def FilterSRTMOD(self,events,maxDepth=50):
        import EQpointSampler_v2 as EQSampler  # optional dependency, not shipped in this repo
        return EQSampler.filter_models_by_location(events,self.lat_min,self.lat_max,self.lon_min,self.lon_max,maxDepth)

    def plot_trenches(self, ax=None, trench_style=None, pb2002_path=None):
        """
        Plots subduction trench boundaries from the PB2002 shapefile.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis on which to plot. If None, a new axis using 
            PlateCarree(central_longitude=180) is created.
        trench_style : dict, optional
            Dictionary of plotting style parameters. Defaults:
                {"color": "red", "linewidth": 2}
        pb2002_path : str
            File path to the PB2002 shapefile.
            
        Returns
        -------
        If a new axis is created, returns (fig, ax); otherwise, returns ax.
        """
        if pb2002_path is None:
            pb2002_path=getFullPath('../subductionData/plateBoundaries/PB2002_boundaries.shp')
        
        if trench_style is None:
            trench_style = {"color": "red", "linewidth": 2}
        
        created_new = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6),
                                   subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            created_new = True
    
        # Load the PB2002 shapefile.
        gdf = gpd.read_file(pb2002_path)
        
        # Filter for subduction boundaries (using the "Type" column).
        trenches = gdf[gdf['Type'].str.contains("Subduction", case=False, na=False)]
        
        # Reproject to EPSG:4326 if needed.
        if trenches.crs is None or trenches.crs.to_string() != "EPSG:4326":
            trenches = trenches.to_crs(epsg=4326)
        
        # Plot the trench boundaries using GeoPandas.
        trenches.plot(ax=ax, transform=ccrs.PlateCarree(), **trench_style)
        
        if created_new:
            return fig, ax
        return ax
    
    
    def plot_all_segments(self, ax, style_params=None):
        if style_params is None:
            style_params = self.coupling_style
        for obj in self.coupling_obj_list:
            obj.plot_segment(ax=ax, style_params=style_params)


    def create_map(self, figsize=(10,8), plotCoupling=True, hillshade=True, ax=None,
                   extent=None):
        """
        extent : [lon_min, lon_max, lat_min, lat_max] optional override.
                 If provided, overrides the plotter's computed boundaries for
                 both the map view and the tick placement.
        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=self.projection)
        else:
            if not isinstance(ax, GeoAxes):
                raise TypeError("Expected ax to be a cartopy GeoAxes (e.g., created with projection=ccrs.PlateCarree()).")
            fig = ax.figure

        # use override extent if provided, otherwise use computed boundaries
        if extent is not None:
            self.lon_min, self.lon_max = extent[0], extent[1]
            self.lat_min, self.lat_max = extent[2], extent[3]

        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max],
                       crs=ccrs.PlateCarree())
        
        ax.set_facecolor('white')
        ax.add_feature(cfeature.OCEAN, facecolor=self.style_params["ocean_color"],
                       alpha=self.style_params["ocean_alpha"])
        ax.add_feature(cfeature.LAND, facecolor=self.style_params["land_color"],
                       alpha=self.style_params["land_alpha"])
        ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black', alpha=0.6)
        
        ax.coastlines(resolution='10m', linewidth=1, color='black')


        if plotCoupling:
            self.plot_all_segments(ax, style_params=self.coupling_style)
        

    # Use the native hillshade extent (from the ETOPo subset) for the image
        if hillshade:
            ax.imshow(self.hillshade, origin='lower',
                      extent=self.hillshade_extent,
                      cmap='Greys',
                      alpha=self.style_params["hillshade_alpha"],
                      transform=ccrs.PlateCarree(),
                      )
        
        #self.plot_trenches(ax=ax)
        


        if not self.style_params.get("hide_ticks", False):
        
            xtick_major = self.style_params["xtick_major"]
            ytick_major = self.style_params["ytick_major"]
            xtick_minor = self.style_params["xtick_minor"]
            ytick_minor = self.style_params["ytick_minor"]

            # use actual ax extent so ticks match even if set_extent is called later
            ext = ax.get_extent(crs=ccrs.PlateCarree())
            cur_lon_min, cur_lon_max = ext[0], ext[1]
            cur_lat_min, cur_lat_max = ext[2], ext[3]

            lon0 = np.floor(cur_lon_min / xtick_major) * xtick_major
            lon1 = np.ceil(cur_lon_max / xtick_major) * xtick_major
        
            lat0 = np.floor(cur_lat_min / ytick_major) * ytick_major
            lat1 = np.ceil(cur_lat_max / ytick_major) * ytick_major
        
            xticks = np.arange(lon0, lon1 + 0.5 * xtick_major, xtick_major)
            yticks = np.arange(lat0, lat1 + 0.5 * ytick_major, ytick_major)
        
            xticks_minor = np.arange(
                np.floor(cur_lon_min / xtick_minor) * xtick_minor,
                np.ceil(cur_lon_max / xtick_minor) * xtick_minor + 0.5 * xtick_minor,
                xtick_minor
            )
        
            yticks_minor = np.arange(
                np.floor(cur_lat_min / ytick_minor) * ytick_minor,
                np.ceil(cur_lat_max / ytick_minor) * ytick_minor + 0.5 * ytick_minor,
                ytick_minor
            )
        
            ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        
            ax.set_xticks(xticks_minor, minor=True, crs=ccrs.PlateCarree())
            ax.set_yticks(yticks_minor, minor=True, crs=ccrs.PlateCarree())
        
            ax.xaxis.set_major_formatter(
                LongitudeFormatter(degree_symbol="°", number_format=".0f")
            )
            ax.yaxis.set_major_formatter(
                LatitudeFormatter(degree_symbol="°", number_format=".0f")
            )
        
            ax.tick_params(
                axis="x", which="major",
                direction="in", bottom=True, top=False,
                length=6, width=1, labelsize=14, pad=4
            )
        
            ax.tick_params(
                axis="y", which="major",
                direction="in", left=True, right=False,
                length=6, width=1, labelsize=14, pad=4
            )
        
            ax.tick_params(
                axis="both", which="minor",
                direction="in", length=3, color="gray", width=0.8
            )
        
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    
        return fig, ax
    
        # if not self.style_params.get("hide_ticks", False):
        #     # create the gridliner
        #     gl = ax.gridlines(
        #         crs=ccrs.PlateCarree(),
        #         draw_labels=True,
        #         linewidth=0.5,
        #         color="gray",
        #         linestyle="--",
        #         alpha=0.5
        #     )
        #     # major tick locations
        #     xt = np.arange(self.lon_min,
        #                    self.lon_max + 1e-6,
        #                    self.style_params["xtick_major"])
        #     yt = np.arange(self.lat_min,
        #                    self.lat_max + 1e-6,
        #                    self.style_params["ytick_major"])
        #     # minor tick locations
        #     xt_minor = np.arange(self.lon_min,
        #                          self.lon_max + 1e-6,
        #                          self.style_params["xtick_minor"])
        #     yt_minor = np.arange(self.lat_min,
        #                          self.lat_max + 1e-6,
        #                          self.style_params["ytick_minor"])
        
        #     # assign locators
        #     gl.xlocator = FixedLocator(xt)
        #     gl.ylocator = FixedLocator(yt)
        #     gl.xminor_locator = FixedLocator(xt_minor)
        #     gl.yminor_locator = FixedLocator(yt_minor)
        
        #     # format labels as °W/°E and °S/°N
        #     gl.xformatter = LongitudeFormatter(degree_symbol="°", number_format=".0f")
        #     gl.yformatter = LatitudeFormatter (degree_symbol="°", number_format=".0f")
        
        #     # disable top/right labels
        #     gl.top_labels = False
        #     gl.right_labels = False
        
        #     # style the label text
        #     gl.xlabel_style = {"size": 22, "weight": "bold"}
        #     gl.ylabel_style = {"size": 22, "weight": "bold"}
        
        # else:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
    
        # return fig, ax
    
    def DistanceFromTrench(self,lon_max=None,lon_min=None,platesToInclude=None):
        import geopandas as gpd
        import closet_distance_to_polygon
        
        if lon_max is None:
            lon_max=self.lon_max
        if lon_min is None:
            lon_min=self.lon_min
        
        boundaries=(lon_min, lon_max, self.lat_min, self.lat_max)
        file=getFullPath("plateBoundaries/PB2002_boundaries.shp")
        plate_boudaries=gpd.read_file(file)
        
        if platesToInclude is None:
           plate_boudaries = plate_boudaries[ plate_boudaries["Type"] == "subduction" ]
        else:
           plate_boudaries = plate_boudaries[plate_boudaries["Name"].isin(platesToInclude)]
        
        distance= closet_distance_to_polygon.DistanceTransformRaster(plate_boudaries, boundaries)
        distance.compute()
        
        return distance
    
    
    def DistanceFromPolygons(self,polygon,lon_max=None,lon_min=None):
        if lon_max is None:
            lon_max=self.lon_max
        if lon_min is None:
            lon_min=self.lon_min
        
        boundaries=(lon_min, lon_max, self.lat_min, self.lat_max)
        
        distance= closet_distance_to_polygon.DistanceTransformRaster(polygon, boundaries)
        distance.compute()
        
        return distance
#%%

def _lon_formatter(x, _):
    """Return 74 °W style labels even if Cartopy gives 286 or -74 etc."""
    # Convert any 0–360 value to the conventional –180…180 range
    if x > 180:
        x -= 360
    hemi = "E" if x > 0 else "W"
    return f"{abs(x):.0f}°{hemi}"


def _lat_formatter(y, _):
    hemi = "S" if y < 0 else "N"
    return f"{abs(y):.0f}°{hemi}"     


#%% these are 

import matplotlib.cm as cm

class Segment:
    """
    Represents a geographic segment defined by a polygon and associated dip and strike angles.
    """
    def __init__(self, polygon: List[Tuple[float, float]], dip: float, strike: float):
        """
        polygon: list of (long, lat) tuples defining the segment boundary.
        dip: dip angle in degrees.
        strike: strike angle in degrees.
        """
        # Path expects (x, y) = (long, lat)
        self._path = Path(polygon)
        self._dip = dip
        self._strike = strike
        self.polygon = polygon

    @property
    def dip_angle(self) -> float:
        """Return the dip angle of the segment (in degrees)."""
        return self._dip

    @property
    def strike_angle(self) -> float:
        """Return the strike angle of the segment (in degrees)."""
        return self._strike

    def contains(self, long: float, lat: float) -> bool:
        """
        Use matplotlib Path to test if point is inside the polygon.
        """
        return self._path.contains_point((long, lat))

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot the polygon boundary on the given Axes (or create one).
        Additional kwargs are passed to plt.plot.
        """
        ax = ax or plt.gca()
        # Extract longitude (x) and latitude (y)
        xs, ys = zip(*(self.polygon + [self.polygon[0]]))
        ax.plot(xs, ys, **kwargs)
        return ax

class SegmentManager:
    """
    Manages multiple geographic segments and dispatches queries for dip/strike based on location.
    """
    def __init__(self, segments: Optional[List[Segment]] = None):
        self.segments: List[Segment] = segments or []

    def add_segment(self, segment: Segment) -> None:
        """Add a new segment to the manager."""
        self.segments.append(segment)

    def get_angles(self, long: float, lat: float) -> Tuple[float, float]:
        """
        Given a point (long, lat), find the containing segment and return its (strike, dip) angles.
        Raises ValueError if the point lies in no segment.
        """
        for seg in self.segments:
            if seg.contains(long, lat):
                return seg.strike_angle, seg.dip_angle
        raise ValueError(f"Point ({long}, {lat}) not contained in any segment")

    def plot_all(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot all segment polygons with distinct colors on the given Axes (or new one).
        """
        ax = ax or plt.subplots()[1]
        num = len(self.segments)
        colormap = cm.get_cmap('tab20', num)
        for idx, seg in enumerate(self.segments):
            color = colormap(idx)
            seg.plot(ax=ax, color=color, label=f"Seg {idx}")
        ax.legend()
        return ax


def load_from_json(filepath: str) -> SegmentManager:
    """
    Load segments from a JSON file and return a populated SegmentManager.

    Expected JSON structure:
    [
      {
        "polygon": [[long1, lat1], [long2, lat2], ...],
        "dip": ..., "strike": ...
      },
      ...
    ]
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    manager = SegmentManager()
    for rec in data:
        seg = Segment(
            polygon=[(pt[0], pt[1]) for pt in rec["polygon"]],
            dip=rec["dip"],
            strike=rec["strike"]
        )
        manager.add_segment(seg)
    return manager