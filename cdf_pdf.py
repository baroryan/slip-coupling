from __future__ import annotations

import time
from typing import Union, Iterable, Tuple, List

import numpy as np
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, progress
from sklearn.neighbors import KernelDensity


ColSpec = Union[str, Tuple[str, ...], List[str]]      # 'x'   or  ('x','y')

#%%
def ecdf_on_grid(samples, x, inclusive=True, force_endpoints=True, weights=None):
    """
    Weighted ECDF evaluated on x.
      samples : 1-D array of observations
      x       : 1-D grid to evaluate ECDF on
      inclusive : if True, uses <= x (right), else < x (left)
      force_endpoints : force y[min(x)]=0 and y[max(x)]=1
      weights : None (uniform), or array same shape as samples (nonnegative)
    """
    s = np.asarray(samples, float)
    x = np.asarray(x, float)

    if weights is None:
        w = np.ones_like(s, dtype=float)
    else:
        w = np.asarray(weights, float)
        if w.shape != s.shape:
            raise ValueError("weights must have same shape as samples")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")

    # keep finite pairs only
    m = np.isfinite(s) & np.isfinite(w)
    s = s[m]; w = w[m]

    if s.size == 0 or np.sum(w) == 0:
        y = np.zeros_like(x, float)
        if force_endpoints and y.size:
            i0, i1 = np.argmin(x), np.argmax(x)
            y[i0], y[i1] = 0.0, 1.0
        return y

    # sort by sample values (stable so ties preserve order)
    idx = np.argsort(s, kind="mergesort")
    s_sorted = s[idx]
    w_sorted = w[idx]

    csum = np.cumsum(w_sorted)
    total = csum[-1]

    side = "right" if inclusive else "left"
    p = np.searchsorted(s_sorted, x, side=side)

    # cumulative weight at each x (<= x for 'right', < x for 'left')
    y = np.where(p > 0, csum[p - 1], 0.0) / total

    if force_endpoints and y.size:
        i0, i1 = np.argmin(x), np.argmax(x)
        y[i0], y[i1] = 0.0, 1.0

    return y
#%%
# ----------------------------------------------------------------------
ColSpec = Union[str, Tuple[str, ...], List[str]]

class ChunkedPDFCDF:
    """
    Chunk wise multivariate KDE (PDF) and 1 D empirical CDF with weights.

    chunk_size : rows per Dask block
    bandwidth  : "scott" | "silverman" | float (isotropic, in scaled space)
    kernel     : kernel accepted by sklearn.neighbors.KernelDensity
    """

    # ------------------ constructor ------------------
    def __init__(
        self,
        *,
        chunk_size: int = 50_000,
        bandwidth: Union[str, float] = "scott",
        kernel: str = "gaussian",
        n_workers: int | None = None,
        client: Client | None = None,
    ):
        self.chunk_size = int(chunk_size)
        self.bandwidth  = bandwidth
        self.kernel     = kernel
        self.n_workers  = n_workers

        if client is not None:
            self._client = client
        elif self.n_workers:
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
                processes=True,
            )
            self._client = Client(cluster)
        else:
            self._client = None

        self._kdes:   List[KernelDensity] = []
        self._sizes:  List[int]           = []
        self._chunk_w_sums: List[float]   = []
        self._n_total: float              = 0.0
        self._scale:  np.ndarray | None   = None  # per axis σ

    # ------------------------------------------------------------------
    # Helpers that run inside tasks
    # ------------------------------------------------------------------
    @staticmethod
    def _n_eff_from_weights(w: np.ndarray) -> float:
        # Kish effective sample size
        w = np.asarray(w, float)
        s1 = np.sum(w)
        s2 = np.sum(w * w)
        if not np.isfinite(s1) or s1 <= 0 or s2 <= 0:
            return 0.0
        return float((s1 * s1) / s2)

    @staticmethod
    def _train_kde(data: np.ndarray, bw: float, kernel: str, w: np.ndarray | None):
        kde = KernelDensity(bandwidth=bw, kernel=kernel)
        try:
            kde.fit(data, sample_weight=w)  # newer sklearn
            return kde
        except TypeError:
            # Fallback: weighted bootstrap to approximate sample_weight
            if w is None:
                kde.fit(data)
                return kde
            w = np.asarray(w, float)
            good = np.isfinite(w) & (w > 0)
            data = data[good]
            w = w[good]
            if data.shape[0] == 0:
                kde.fit(np.zeros((1, data.shape[1])))
                return kde
            n_eff = int(max(len(data), round(ChunkedPDFCDF._n_eff_from_weights(w))))
            p = w / w.sum()
            rng = np.random.default_rng(12345)
            idx = rng.choice(len(data), size=n_eff, replace=True, p=p)
            kde.fit(data[idx])
            return kde

    @staticmethod
    def _chunk_cumweights(data: np.ndarray, x_eval: np.ndarray, w: np.ndarray | None):
        """Return cumulative weights for values <= x_eval for a 1 D array."""
        data = np.asarray(data, float)
        finite = np.isfinite(data)
        vals = data[finite]
        if vals.size == 0:
            return np.zeros_like(x_eval, float)
        if w is None:
            wv = np.ones_like(vals, float)
        else:
            w = np.asarray(w, float)
            wv = w[finite]
        idx = np.argsort(vals, kind="mergesort")
        vs = vals[idx]
        ws = wv[idx]
        csum = np.cumsum(ws)
        p = np.searchsorted(vs, x_eval, side="right")
        return np.where(p > 0, csum[p - 1], 0.0)

    @staticmethod
    def _score_chunk(kde: KernelDensity, pts_scaled: np.ndarray, chunk_weight_sum: float):
        # score_samples returns density in scaled space; scale by chunk "mass"
        return np.exp(kde.score_samples(pts_scaled)) * float(chunk_weight_sum)

    @staticmethod
    def _flatten_delayed(nested):
        if isinstance(nested, (list, tuple)):
            out = []
            for item in nested:
                out.extend(ChunkedPDFCDF._flatten_delayed(item))
            return out
        elif hasattr(nested, "ravel"):
            return list(nested.ravel())
        else:
            return [nested]

    # ------------------------------------------------------------------
    # Data → Dask array utilities
    # ------------------------------------------------------------------
    def _to_dask_array(self, data, column: str):
        if hasattr(data, "columns"):
            series = data[column]
        else:
            series = data
        if hasattr(series, "to_dask_array"):
            darr = series.to_dask_array()
        else:
            arr  = getattr(series, "values", series)
            darr = da.from_array(np.asarray(arr), chunks=(self.chunk_size,))
        return darr

    def _to_dask_matrix(self, data, columns: ColSpec) -> da.Array:
        if isinstance(columns, (list, tuple)):
            cols = list(columns)
            if hasattr(data, "to_dask_array"):
                return data[cols].to_dask_array()
            if hasattr(data, "loc"):
                arr = data.loc[:, cols].to_numpy(float, copy=False)
                return da.from_array(arr, chunks=(self.chunk_size, len(cols)))
            arr = np.asarray(data, float)
            if arr.ndim != 2 or arr.shape[1] != len(cols):
                raise ValueError("Provide a DataFrame or 2 D array with matching columns.")
            return da.from_array(arr, chunks=(self.chunk_size, len(cols)))
        darr1d = self._to_dask_array(data, columns)
        return darr1d[:, None]

    # ------------------------------------------------------------------
    # Global bandwidth in scaled space, with optional n_eff
    # ------------------------------------------------------------------
    def _global_bandwidth(self, darr_scaled: da.Array, n_eff: float | None = None) -> float:
        n, d = darr_scaled.shape
        n0 = float(n_eff) if n_eff is not None and n_eff > 0 else float(n)
        if self.bandwidth == "scott":
            return n0 ** (-1 / (d + 4))
        if self.bandwidth == "silverman":
            return 0.9 * n0 ** (-1 / (d + 4))
        return float(self.bandwidth)

    # ------------------------------------------------------------------
    # --------------------------- FIT ---------------------------
    # ------------------------------------------------------------------
    def fit(self, data, column: ColSpec, weights: Iterable[float] | None = None):
        """
        Train one KDE per chunk. If weights is provided, it must be 1D of length N.
        Uses weighted per-axis scaling, n_eff for bandwidth, and chunk-weight mixing.
        """
        print(f"Starting fit: chunk_size={self.chunk_size}")
    
        darr = self._to_dask_matrix(data, column).compute_chunk_sizes()
        darr = darr.rechunk((self.chunk_size, -1))
        n_samples, n_dims = darr.shape
    
        # weights to dask
        if weights is not None:
            w_arr = np.asarray(weights, float).ravel()
            if w_arr.size != n_samples:
                raise ValueError("weights length must match number of rows")
            w_da = da.from_array(w_arr, chunks=(self.chunk_size,))
        else:
            w_da = None
    
        # ----- weighted scaling (key fix) -----
        if w_da is None:
            # unweighted mean and std
            mean = darr.mean(axis=0)
            var  = darr.var(axis=0)
        else:
            # per-axis weighted mean and var
            # mean = sum(w x) / sum(w)
            wsum = w_da.sum()
            wx   = da.einsum("n,nd->d", w_da, darr)
            mean = wx / wsum
            # var = sum(w (x-mean)^2) / sum(w)
            xc   = darr - mean  # broadcasts over rows
            wxc2 = da.einsum("n,nd->d", w_da, xc * xc)
            var  = wxc2 / wsum
    
        self._scale = np.sqrt(var.compute()).astype(float)
        self._scale[self._scale == 0] = 1.0
        darr_scaled = darr / self._scale
    
        # per‑chunk weight sums
        self._sizes = [int(s) for s in darr_scaled.chunks[0]]
        if w_da is None:
            self._chunk_w_sums = [float(s) for s in self._sizes]
            n_eff_global = float(n_samples)
        else:
            w_blocks = self._flatten_delayed(w_da.to_delayed())
            with ProgressBar():
                chunk_sums = compute(*[delayed(np.sum)(wb) for wb in w_blocks])
            self._chunk_w_sums = [float(v) for v in chunk_sums]
            n_eff_global = self._n_eff_from_weights(w_da.compute().astype(float))
    
        self._n_total = float(sum(self._chunk_w_sums))
        print(f"Total chunks: {len(self._sizes)}   total weight: {self._n_total:g}")
    
        # bandwidth in scaled space using n_eff
        bw = self._global_bandwidth(darr_scaled, n_eff=n_eff_global)
        print(f"Bandwidth (scaled space): {bw:.4g}")
    
        # build training tasks
        blocks = self._flatten_delayed(darr_scaled.to_delayed())
        if w_da is None:
            w_blocks = [None] * len(blocks)
        else:
            w_blocks = self._flatten_delayed(w_da.to_delayed())
    
        tasks = [delayed(self._train_kde)(blk, bw, self.kernel, wblk)
                 for blk, wblk in zip(blocks, w_blocks)]
    
        # execute
        if self._client:
            print("Distributed fit with progress")
            futures = self._client.compute(tasks)
            progress(futures)
            self._kdes = self._client.gather(futures)
        else:
            print("Local fit with ProgressBar")
            with ProgressBar():
                self._kdes = compute(*tasks)
            self._kdes = list(self._kdes)
    
        print(f"Finished fit: {len(self._kdes)} KDE models.")

    # ------------------------------------------------------------------
    # --------------------------- PDF ---------------------------
    # ------------------------------------------------------------------
    def pdf(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted PDF at M query points. Uses chunk weight sums for mixing.
        """
        if not self._kdes:
            raise RuntimeError("fit() must be called before pdf().")

        pts = np.asarray(pts, float)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.shape[1] != len(self._scale):
            raise ValueError(f"Expected {len(self._scale)} D points, got {pts.shape[1]} D.")

        pts_scaled = pts / self._scale
        tasks = [delayed(self._score_chunk)(kde, pts_scaled, wsum)
                 for kde, wsum in zip(self._kdes, self._chunk_w_sums)]

        if self._client:
            futures = self._client.compute(tasks)
            progress(futures)
            parts = self._client.gather(futures)
        else:
            with ProgressBar():
                parts = compute(*tasks)

        pdf_sum = sum(parts)
        return (pdf_sum / float(self._n_total) / np.prod(self._scale)).ravel()

    # Convenience: evaluate on a 2 D grid and reshape
    def pdf2d(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        if len(self._scale) != 2:
            raise RuntimeError("pdf2d is only available after fitting 2 D data.")
        pts = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        z = self.pdf(pts)
        return z.reshape(x_grid.shape)

    # ------------------------------------------------------------------
    # --------------------------- CDF (1 D) ---------------------------
    # ------------------------------------------------------------------
    def cdf(self, x: np.ndarray, data, column: str, weights: Iterable[float] | None = None) -> np.ndarray:
        """
        Weighted empirical CDF of one column.
        weights is 1 D with length N, or None for unweighted.
        """
        if isinstance(column, (list, tuple)):
            raise ValueError("cdf() expects a single column name for 1 D CDF.")

        x_arr = np.asarray(x).ravel()
        darr  = self._to_dask_array(data, column).compute_chunk_sizes()
        darr  = darr.rechunk(self.chunk_size)

        if weights is not None:
            w_arr = np.asarray(weights, float).ravel()
            if w_arr.size != darr.size:
                raise ValueError("weights length must match number of rows")
            w_da = da.from_array(w_arr, chunks=(self.chunk_size,))
            w_blocks = self._flatten_delayed(w_da.to_delayed())
            total_w = float(np.sum(w_arr))
            if not np.isfinite(total_w) or total_w <= 0:
                raise ValueError("Sum of weights must be positive and finite")
        else:
            w_blocks = [None] * len(self._flatten_delayed(darr.to_delayed()))
            total_w = float(darr.size)

        data_blocks = self._flatten_delayed(darr.to_delayed())
        tasks = [delayed(self._chunk_cumweights)(blk, x_arr, wblk)
                 for blk, wblk in zip(data_blocks, w_blocks)]

        if self._client:
            futures = self._client.compute(tasks)
            progress(futures)
            cumw_parts = self._client.gather(futures)
        else:
            with ProgressBar():
                cumw_parts = compute(*tasks)

        cumw = np.sum(cumw_parts, axis=0)
        return (cumw / total_w).astype(float)

    # ------------------------------------------------------------------
    # --------------------------- Wrapper ---------------------------
    # ------------------------------------------------------------------
    def pdf_cdf(self, x: np.ndarray, data, column: str, cdf_weights: Iterable[float] | None = None):
        print("Starting CDF computation...", flush=True)
        cdf = self.cdf(x, data, column, weights=cdf_weights)
        print("Finished CDF computation.", flush=True)

        print("Starting PDF computation...", flush=True)
        pdf = self.pdf(x)
        print("Finished PDF computation.", flush=True)
        return pdf, cdf
#%%


def counts_block(block, mask_block, x_vals, weights_block=None):
    """
    Return, for each column j, the cumulative *weight* of values <= x for x in x_vals.
    If weights_block is None → each observation has weight 1.
    If weights_block has shape (n_rows,) → same row-weight for all columns.
    If weights_block has shape (n_rows, k) → per-element weights (column-specific).
    """
    m = np.asarray(mask_block, dtype=bool).reshape(-1)
    assert m.shape[0] == block.shape[0]

    B = np.ascontiguousarray(block[m, :])     # (rows_in_block, k)
    k = B.shape[1]
    M = x_vals.size

    out = np.zeros((k, M), dtype=np.float64)
    if B.size == 0:
        return out

    # Normalize weights to a 2D array aligned with B
    if weights_block is None:
        W = np.ones_like(B, dtype=np.float64)
    else:
        wb = weights_block[m]
        if wb.ndim == 1:
            W = np.broadcast_to(wb[:, None], B.shape).astype(np.float64, copy=False)
        elif wb.ndim == 2:
            assert wb.shape == B.shape
            W = wb.astype(np.float64, copy=False)
        else:
            raise ValueError("weights_block must be None, (rows,), or (rows, k)")

    # Per-column: sort by values, carry weights, do prefix-sum and search
    isfinite_B = np.isfinite(B)
    for j in range(k):
        finite_mask = isfinite_B[:, j]
        if not np.any(finite_mask):
            continue

        v = B[finite_mask, j]
        w = W[finite_mask, j]

        # sort values and weights together
        idx = np.argsort(v, kind="mergesort")
        v_sorted = v[idx]
        w_sorted = w[idx]

        csum = np.cumsum(w_sorted)  # prefix sums of weights
        # positions for each x in the sorted array (right → <= x)
        p = np.searchsorted(v_sorted, x_vals, side="right")

        # cumulative weight at x is csum[p-1] when p>0 else 0
        out[j] = np.where(p > 0, csum[p - 1], 0.0)
        
        

    return out  # (k, len(x_vals)), cumulative *weights* (not normalized)
#%%
import dask

import xarray as xr

class CDF_per_columns:
    def __init__(self, data, x, max_k=None,weights=None, chunk_rows=20000):
        # normalize x
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size == 0:
            raise ValueError("x must have at least one value")
        if not np.all(np.diff(x) > 0):
            x = np.unique(x.astype(np.float64)).astype(np.float32)
        self.x = x
        self.chunk_rows=chunk_rows
        arr_da = self.to_dask_2d(data, max_k=max_k)
        self.da = arr_da
        self.N, self.k = map(int, arr_da.shape)
        
        
        self.weights_da=self.take_care_of_weights(weights)
        
    def dd_series_to_bool_da(self,series, row_chunks):
        s = series.astype("bool")
        try:
            arr = s.to_dask_array(lengths=True)
        except TypeError:
            arr = s.to_dask_array()
            arr = arr.compute_chunk_sizes()
        return arr.rechunk(row_chunks)
        
        
    def to_dask_2d(self,data, *,  max_k=None) -> da.Array:
        """
        Return a 2D Dask array (N, k) from xarray, NumPy, or Dask input.
        If input is 1D, it is promoted to shape (N, 1).
        """
        # unwrap xarray
        chunk_rows=self.chunk_rows
            
        if isinstance(data, xr.DataArray):
            data = data.data  # could be NumPy or Dask
    
        # normalize to Dask array
        if isinstance(data, da.Array):
            arr_da = data
        elif isinstance(data, np.ndarray):
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim != 2:
                raise ValueError("data must be 1D or 2D")
            arr_da = da.from_array(arr, chunks=(chunk_rows, arr.shape[1]))
        else:
            raise TypeError("data must be numpy.ndarray, dask.array.Array, or xarray.DataArray")
    
        # ensure 2D
        if arr_da.ndim == 1:
            arr_da = arr_da[:, None]
        elif arr_da.ndim != 2:
            raise ValueError("data must be 1D or 2D")
    
        # optional column cap
        if max_k is not None:
            arr_da = arr_da[:, :int(max_k)]
    
        # align chunks: rows = chunk_rows, single column chunk
        arr_da = arr_da.rechunk((chunk_rows, -1))
        return arr_da
            
    def take_care_of_weights(self, weights):
        """Return a Dask array aligned with self.da, or None."""
        if weights is None:
            weights_da = None
    
        elif isinstance(weights, xr.DataArray):
            arr = weights.data  # could be NumPy or Dask
            if isinstance(arr, da.Array):
                weights_da = arr
            elif isinstance(arr, np.ndarray):
                if arr.ndim == 1:
                    weights_da = da.from_array(arr, chunks=self.da.chunks[0])
                elif arr.ndim == 2:
                    if arr.shape != self.da.shape:
                        raise ValueError("weights with shape (N,k) must match data shape")
                    weights_da = da.from_array(arr, chunks=self.da.chunks)
                else:
                    raise ValueError("weights must be 1D or 2D")
            else:
                raise TypeError("Unsupported xarray.data backing array for weights")
    
        elif isinstance(weights, np.ndarray):
            if weights.ndim == 1:
                weights_da = da.from_array(weights, chunks=self.da.chunks[0])
            elif weights.ndim == 2:
                if weights.shape != self.da.shape:
                    raise ValueError("weights with shape (N,k) must match data shape")
                weights_da = da.from_array(weights, chunks=self.da.chunks)
            else:
                raise ValueError("weights must be 1D or 2D")
    
        elif isinstance(weights, da.Array):
            weights_da = weights
    
        else:
            raise TypeError("weights must be None, numpy, dask.array, or xarray.DataArray")
    
        # Validate and align chunks
        if weights_da is not None:
            # basic validation
            if weights_da.ndim == 1 and weights_da.shape[0] != self.N:
                raise ValueError(f"1D weights length {weights_da.shape[0]} must equal N={self.N}")
            if weights_da.ndim == 2 and weights_da.shape != self.da.shape:
                raise ValueError(f"2D weights shape {weights_da.shape} must equal data shape {self.da.shape}")
            # finite and nonnegative
            if da.any(~da.isfinite(weights_da)).compute():
                raise ValueError("weights contain nonfinite values")
            if da.any(weights_da < 0).compute():
                raise ValueError("weights must be nonnegative")
            # rechunk to match data
            weights_da = weights_da.rechunk(self.da.chunks[0] if weights_da.ndim == 1 else self.da.chunks)
    
        return weights_da
    def _series_to_numpy_bool(self,s):
        # top-level function so it's picklable for dask.delayed
        return np.asarray(s, dtype=bool)

    def computeCDFsForSubset(self, mask=None, returnXarray=False, progress: str = "bar"):
        """
        Compute ECDFs at self.x for each column.
        Uses self.weights_da (set in __init__) if present.
    
        progress:
            "none"   -> no progress output
            "bar"    -> dask ProgressBar
            "chunks" -> prints 'done i/n_blocks' as each block finishes (requires distributed Client)
        """
        start_time = time.time()
    
        # ---------- mask -> dask.bool ----------
        if mask is None:
            mask_da = da.ones((self.N,), dtype=bool, chunks=self.da.chunks[0])
        elif isinstance(mask, xr.DataArray):
            mask_da = da.from_array(mask.data.astype(bool), chunks=self.da.chunks[0])
        elif isinstance(mask, np.ndarray):
            mask_da = da.from_array(mask.astype(bool), chunks=self.da.chunks[0])
        elif isinstance(mask, da.Array):
            mask_da = mask.rechunk(self.da.chunks[0])
        elif hasattr(mask, "to_delayed"):  # dask.dataframe.Series (expr or classic)
            # ensure boolean Series (avoid bool[pyarrow] quirks)
            s = mask.astype("bool")
        
            # 1) get one delayed pandas Series per partition
            parts = s.to_delayed()  # list[delayed[pd.Series]]
        
            # 2) compute partition lengths deterministically
            part_lens = dask.compute(*[dask.delayed(len)(p) for p in parts])
        
            # 3) wrap each partition as a 1-D dask.array block with known shape
            arr_blocks = [da.from_delayed(dask.delayed(self._series_to_numpy_bool)(p),shape=(int(n),),dtype=bool,)
                for p, n in zip(parts, part_lens)
            ]
        
            # 4) concatenate into a single 1-D mask array
            mask_da = da.concatenate(arr_blocks, axis=0)
        
            # 5) align row chunks with data
            mask_da = mask_da.rechunk(self.da.chunks[0])
        else:
            raise TypeError("mask must be None, numpy, dask.array, or xarray.DataArray")
    
        # ---------- weights: use self.weights_da by default ----------
        weights_da = self.weights_da
        if weights_da is not None:
            if weights_da.ndim == 1:
                weights_da = weights_da.rechunk(self.da.chunks[0])
            else:
                weights_da = weights_da.rechunk(self.da.chunks)
    
        # ---------- delayed blocks ----------
        data_blocks = self.da.to_delayed().ravel()
        mask_blocks = mask_da.to_delayed().ravel()
        if weights_da is None:
            weights_blocks = [None] * len(data_blocks)
        else:
            weights_blocks = weights_da.to_delayed().ravel()
    
        delayed_counts = [
            dask.delayed(counts_block)(db, mb, self.x, wb)
            for db, mb, wb in zip(data_blocks, mask_blocks, weights_blocks)
        ]
        n_blocks = len(delayed_counts)
    
        # Evaluate per-block counts with optional chunk progress
        if progress == "chunks" and getattr(self, "_client", None) is not None:
            from dask.distributed import as_completed
            futures = self._client.compute(delayed_counts)
            parts = [None] * n_blocks
            done = 0
            for f in as_completed(futures):
                idx = futures.index(f)
                parts[idx] = f.result()
                done += 1
                print(f"blocks done {done}/{n_blocks}", flush=True)
            counts_blocks = da.stack(
                [da.from_array(part, chunks=(self.k, self.x.size)) for part in parts],
                axis=0
            )
        else:
            # compute all blocks at once (optionally with a bar)
            if progress == "bar":
                from dask.diagnostics import ProgressBar
                with ProgressBar():
                    parts = dask.compute(*delayed_counts)
            else:
                parts = dask.compute(*delayed_counts)
            counts_blocks = da.stack(
                [da.from_array(part, chunks=(self.k, self.x.size)) for part in parts],
                axis=0
            )
    
        # ---------- aggregate cumulative weights ----------
        cumw_le_x = counts_blocks.sum(axis=0)  # (k, len(x))
    
        # ---------- totals per column for normalization ----------
        finite = da.isfinite(self.da)
        if weights_da is None:
            n_valid = da.sum(finite & mask_da[:, None], axis=0).astype(np.float64)
        else:
            if weights_da.ndim == 1:
                n_valid = da.sum(finite * weights_da[:, None] * mask_da[:, None], axis=0)
            else:
                n_valid = da.sum(finite * weights_da * mask_da[:, None], axis=0)
    
        n_valid = da.maximum(n_valid, 1.0)
        CDF = (cumw_le_x / n_valid[:, None]).astype(np.float32)  # (k, len(x))
    
        # final compute
        if progress == "bar":
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                CDF_np = dask.compute(CDF)[0]
        else:
            CDF_np = dask.compute(CDF)[0]
    
        if returnXarray:
            cdf_da = xr.DataArray(
                CDF_np.T,
                dims=("x", "k"),
                coords={"x": self.x, "k": np.arange(self.k, dtype=np.int32)},
                name="cdf",
            )
        else:
            cdf_da = np.asarray(CDF_np.T, dtype=np.float32, order="C")
    
        total_rows = int(mask_da.sum().compute())
        print(f"Processed ~{total_rows} rows x {self.k} cols in {time.time()-start_time:.2f}s", flush=True)
        return cdf_da
    
#%%
import sys
def _p_superiority(h, n):
    """P{H>N} for one x, using all pairs via searchsorted."""
    h = np.sort(h); n = np.sort(n)
    return np.searchsorted(n, h, side='right').sum() / (h.size * n.size)

def pgt_with_ci_fast(H, N, B=1000, seed=None, progress=True, every=50, alpha=0.05):
    """
    H, N: arrays (m_x, nH), (m_x, nN)
    Returns: p_gt, ci_low, ci_high  (each shape (m_x,))
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    m_x, nH = H.shape
    _,   nN = N.shape

    p_gt   = np.empty(m_x, dtype=float)
    ci_low = np.empty(m_x, dtype=float)
    ci_high= np.empty(m_x, dtype=float)

    for k in range(m_x):
        h = H[k]; n = N[k]
        # point estimate from all pairs
        est = _p_superiority(h, n)
        p_gt[k] = est

        # bootstrap (resample the 500 H's and 500 N's, not the 250k pairs)
        boots = np.empty(B, dtype=float)
        for b in range(B):
            hs = h[rng.integers(0, nH, nH)]
            ns = n[rng.integers(0, nN, nN)]
            boots[b] = _p_superiority(hs, ns)

        lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
        ci_low[k], ci_high[k] = lo, hi

        if progress and ((k+1) % every == 0 or (k+1) == m_x):
            elapsed = time.time() - t0
            done = k + 1
            rate = done / max(elapsed, 1e-9)
            remain = (m_x - done) / rate if rate > 0 else float('inf')
            sys.stdout.write(
                f"\r[{done}/{m_x}]  elapsed: {elapsed:6.1f}s  "
                f"eta: {remain:6.1f}s  avg {rate:5.1f} x/s"
            )
            sys.stdout.flush()
    if progress:
        sys.stdout.write("\n")
    return p_gt, ci_low, ci_high