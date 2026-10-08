#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:56:56 2025

@author: bar
"""




import numpy as np
import pandas as pd
import finiteFaultSampler as ffs
import re
from dateutil.parser import parse
from io import StringIO
from pathlib import Path

        
#%%
# ------------- helpers ------------------------------------------------
LATLON_HDR_RE   = re.compile(r'%\s+LAT\s+LON', re.I)

DATE_RE = re.compile(                   # YYYY/MM/DD  or  MM/DD/YYYY (or DD/MM/YYYY)
    r'\b(?:\d{4}[/-]\d{2}[/-]\d{2}|'    # 2012/08/27
    r'\d{2}[/-]\d{2}[/-]\d{4})\b' ,     # 11/14/2007
    re.A
)

DX_RE  = re.compile(r'Dx\s*=\s*([0-9.+-eE]+)', re.I)
DZ_RE  = re.compile(r'Dz\s*=\s*([0-9.+-eE]+)', re.I)
BRACKET_REF_RE = re.compile(r'\[([^\]]+?)\]')        # stuff inside [...]

EVENT_HDR_RE = re.compile(r'%\s*Event\b', re.I)      # "%Event", "%  Event", etc.

def _find_dx_dz(line: str):
    mdx, mdz = DX_RE.search(line), DZ_RE.search(line)
    return (float(mdx.group(1)), float(mdz.group(1))) if mdx and mdz else (None, None)

def _parse_event_line(line: str, meta: dict) -> bool:
    """
    Extracts:
        • meta["Date"]  (string, as found)
        • meta["Date_parsed"]  (datetime.date)
        • meta["Event"] (event name before the date)
        • meta["Reference"] (contents of […] block, compressed)
    Returns True if a date was found, else False.
    """
    payload = line.split(":", 1)[1] if ":" in line else line   # after first colon
    text = payload.strip()

    # -------- reference in […] ----------------------------------------
    ref_match = BRACKET_REF_RE.search(text)
    if ref_match:
        ref_raw = ref_match.group(1)
        ref_clean = " ".join(re.sub(r'[()]', '', ref_raw).split())  # drop ( ) + tidy spaces
        meta["Reference"] = ref_clean
        # remove the bracket part from the working text so it can’t confuse date parsing
        text = text[:ref_match.start()] + text[ref_match.end():]

    # -------- explicit date regex -------------------------------------
    m = DATE_RE.search(text)
    if m:
        date_str = m.group(0)
        meta["Date"] = date_str
        try:
            meta["Date_parsed"] = parse(date_str, dayfirst=False).date()
        except Exception:
            pass
        # everything before the match = event name
        event = text[:m.start()].strip()
        event = re.sub(r'[,\-\s]+$', '', event)          # trim trailing , - space
        meta["Event"] = event or None
        return True

    # -------- fuzzy fallback (rare) -----------------------------------
    try:
        dt = parse(text, fuzzy=True)
        meta["Date_parsed"] = dt.date()
        meta["Date"] = dt.strftime("%Y/%m/%d")
        meta["Event"] = text.split(meta["Date"])[0].strip(" ,-\t")
        return True
    except Exception:
        return False
# ----------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: robust reader for the numeric block of one segment
# ---------------------------------------------------------------------------
def _read_numeric_block(block: str) -> pd.DataFrame:
    """
    Parse a numeric block of an SRCMOD/USGS *.fsp* segment without assuming a
    fixed column count.

    The first eight columns are fixed across all known formats:
        c0  LAT
        c1  LON
        c2  X  (east–west, km)
        c3  Y  (north–south, km)
        c4  Z  (depth, km – positive down)
        c5  SLIP  (m)
        c6  RAKE  (deg)     [missing in a handful of legacy files]
        c7  TRUP  (s)

    Any remaining columns (TW1, rakeTW1, TW2, …) are kept but not used here.
    """
    df = pd.read_csv(
        StringIO(block),
        header=None,               # no header row in the numeric block
        sep='\s+',     # arbitrary spacing
        engine="python",           # more forgiving than the C parser
    )
    # Give the columns generic names: c0, c1, … cN
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df




# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def ReadFSPFile(filename, extractWebpage: bool = True):
    """
    Parse a USGS/SRCMOD *.fsp finite-fault file (single- or multi-segment).

    Returns
    -------
    ffs.finiteFault
        .meta     – global metadata (dict)
        .segments – list[ffs.LatLonFaultSegment]
    """
    segments, meta, data_dict = [], {}, None
    global_dx = global_dz = None          # default sub-fault dimensions
    meta["Webpage"] = "SRCMOD"

    # ---------- 1) read whole file ------------------------------------------
    with open(filename, "r") as f:
        lines = f.readlines()

    if extractWebpage:
        meta["Webpage"] = lines[2].strip()

    # ---------- 2) scan header / comment lines ------------------------------
    date_found = False
    for i, line in enumerate(lines):

        # --- event location --------------------------------------------------
        if line.startswith("% Loc"):
            mlat = re.search(r"LAT\s*=\s*([0-9.+-eE]+)", line)
            mlon = re.search(r"LON\s*=\s*([0-9.+-eE]+)", line)
            mdep = re.search(r"DEP\s*=\s*([0-9.+-eE]+)", line)
            if mlat: meta["lat"]   = float(mlat.group(1))
            if mlon: meta["long"]  = float(mlon.group(1))
            if mdep: meta["depth"] = float(mdep.group(1))

        # --- “% Data …” lines (labels + counts) -----------------------------
        elif line.startswith("% Data") and data_dict is None:
            labels = re.split(r"\s+", line.split(":", 1)[1].strip())
            if i + 1 < len(lines) and lines[i+1].startswith(("% Data", "% NoS")):
                counts_line = lines[i + 1]
                counts = [int(x) for x in re.split(r"\s+", counts_line.split(":", 1)[1].strip()) if x]
                data_dict = dict(zip(labels, counts))
                meta["data"] = data_dict

        # --- inversion summary ----------------------------------------------
        elif line.startswith("% Invs"):
            # Nsg (segments)
            m_nsg = re.search(r"Nsg\s*=\s*(\d+)", line)
            if m_nsg: meta["Nsg"] = int(m_nsg.group(1))
            dx, dz = _find_dx_dz(line)
            if dx is not None:
                global_dx, global_dz = dx, dz
                meta["size_LEN"], meta["size_WID"] = dx, dz

        # --- overall size line (still contains Mw) ---------------------------
        elif line.startswith("% Size"):
            parts = line.split()
            meta["Mw"] = float(parts[13])

        elif line.startswith("% EventTAG:"):
            meta["EventTAG"] = line.split(":", 1)[1].strip()

        elif line.startswith("% Event :"):
            date_found = _parse_event_line(line, meta)

        # --- global strike/dip ----------------------------------------------
        elif line.startswith("% Mech"):
            mstk = re.search(r"STRK\s*=\s*([0-9.+-eE]+)", line)
            mdip = re.search(r"DIP\s*=\s*([0-9.+-eE]+)",  line)
            if mstk: meta["global_strike"] = float(mstk.group(1))
            if mdip: meta["global_dip"]    = float(mdip.group(1))

        meta["invAUTH"] = "USGS"

    # -------- ensure we got a Date ------------------------------------------
    if not date_found or "Date" not in meta:
        raise ValueError("Failed to extract a date from the '% Event :' line "
                         f"in {Path(filename).name}. This file is malformed "
                         "or the parser needs further adjustment.")

    # ---------- 3) segment parsing ------------------------------------------
    seg_headers = [idx for idx, L in enumerate(lines) if L.startswith("% SEGMENT")]

    def _make_segment(start_idx, strike, dip, length, width):
        """Create one LatLonFaultSegment from the block that starts at *start_idx*."""
        # find the LAT LON header and grab the column names
        for k in range(start_idx, len(lines)):
            if LATLON_HDR_RE.match(lines[k]):
                header_line = lines[k].strip().lstrip("%").strip()
                header_tokens = re.split(r"\s+", header_line)
                data_start = k + 2  # header + dashed separator
                break
        else:
            raise ValueError("Could not find data table for segment.")
    
        # accumulate numeric rows
        rows = []
        while data_start < len(lines) and not lines[data_start].startswith("%"):
            rows.append(lines[data_start])
            data_start += 1
    
        df = _read_numeric_block("".join(rows))
    
        # base columns (robust to legacy files without RAKE)
        lat   = df["c0"].values
        lon   = df["c1"].values
        depth = df["c4"].values
        slip  = df["c5"].values
        rake  = df["c6"].values if "c6" in df.columns else np.zeros_like(slip)
    
        # figure out where "extra" columns start: after RAKE if present, else after SLIP
        def _idx_of(name):
            for i, t in enumerate(header_tokens):
                if t.upper() == name:
                    return i
            return None
    
        start_extra = _idx_of("RAKE")
        if start_extra is None:
            start_extra = _idx_of("SLIP")
        start_extra = (start_extra + 1) if start_extra is not None else df.shape[1]  # none -> no extras
    
        extra_cols = {}
        for i in range(start_extra, df.shape[1]):
            colname = header_tokens[i] if i < len(header_tokens) else f"c{i}"
            extra_cols[colname] = df[f"c{i}"].values
    
        additional_df = pd.DataFrame(extra_cols) if extra_cols else pd.DataFrame({})
    
        seg = ffs.LatLonFaultSegment(
            slip=slip,
            x=lon,
            y=lat,
            z=depth,
            rake=rake,
            strike=float(strike),
            dip=float(dip),
            DimWL=np.array([length, width]),
            additionalData=additional_df,
        )

    
        return seg

    # --- A) MULTI-segment file ----------------------------------------------
    if seg_headers:
        seg_headers.append(len(lines))     # sentinel at EOF
        for h, idx in enumerate(seg_headers[:-1]):
            hdr    = lines[idx]
            strike = float(re.search(r"STRIKE\s*=\s*([0-9.+-eE]+)", hdr).group(1))
            dip    = float(re.search(r"DIP\s*=\s*([0-9.+-eE]+)",    hdr).group(1))

            # default dims from global Dx/Dz
            width, length = global_dx, global_dz

            # look for per-segment Dx/Dz overrides
            for j in range(idx, seg_headers[h + 1]):
                dx, dz = _find_dx_dz(lines[j])
                if dx is not None:
                    width, length = dx, dz
                    break

            # final fallback to legacy LEN/WID line
            if width is None or length is None:
                size_line = lines[idx + 1]
                m_len = re.search(r"LEN\s*=\s*([0-9.+-eE]+)", size_line)
                m_wid = re.search(r"WID\s*=\s*([0-9.+-eE]+)", size_line)
                if m_len and m_wid:
                    width, length = float(m_len.group(1)), float(m_wid.group(1))
                else:
                    raise ValueError("No Dx/Dz or LEN/WID found for segment.")

            segments.append(_make_segment(idx, strike, dip, length, width))

    # --- B) SINGLE-segment file ---------------------------------------------
    else:
        # locate data table header
        for k, L in enumerate(lines):
            if LATLON_HDR_RE.match(L):
                data_start_idx = k
                break
        else:
            raise ValueError("No data table found for single-segment file.")

        strike = meta.get("global_strike")
        dip    = meta.get("global_dip")
        width  = global_dx if global_dx is not None else meta.get("size_LEN")
        length = global_dz if global_dz is not None else meta.get("size_WID")
        if width is None or length is None:
            raise ValueError("Could not determine Dx/Dz (width/length).")

        segments.append(_make_segment(data_start_idx, strike, dip, length, width))

    # ---------- 4) wrap up ---------------------------------------------------
    return ffs.finiteFault(meta, segments)