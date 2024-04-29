import os
import pystac
import sys

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from pystac_client import Client
from eof.download import download_eofs
from rasterio.features import rasterize
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import datetime, timedelta

from helpers import load_poly_from_geojson

EPOCH_SENTINEL1 = datetime.strptime("20140414", "%Y%m%d")


def sentinel1_acquired(
    download_dir: str,
    country_name: str,
    stac_items: pystac.ItemCollection,
    save_to: str = None,
):
    """Return a map showing the footprint of the Sentinel-1 acquisition for a set of stac items"""
    base_dir = os.path.join(download_dir, country_name)
    country_geom = load_poly_from_geojson(
        os.path.join(base_dir, f"{country_name}.geojson")
    )

    base_grid = xr.open_dataarray(
        os.path.join(base_dir, f"{country_name}_grid.tif"), engine="rasterio"
    )

    footprint = np.zeros(tuple(base_grid.rio.shape), dtype=float)

    for item in reversed(list(stac_items)):
        current_acq = np.datetime64(
            datetime.strptime(
                item.properties["start_datetime"], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        )

        mask = rasterize(
            [item.geometry],
            base_grid.rio.shape,
            transform=base_grid.rio.transform(),
            all_touched=True,
        )

        footprint[mask > 0] = 1

    footprint_da = xr.DataArray(
        footprint,
        dims=("y", "x"),
        coords={
            "y": base_grid.y,
            "x": base_grid.x,
        },
    )
    footprint_da.attrs = base_grid.attrs
    footprint_da.rio.write_crs("EPSG:4326", inplace=True)
    footprint_da.rio.write_nodata(
        -sys.maxsize - 1,
        encoded=True,
        inplace=True,
    )

    footprint_da = footprint_da.rio.clip(
        [country_geom], all_touched=True, from_disk=True
    )

    if save_to:
        footprint_da.rio.to_raster(save_to)

    return footprint_da


def create_sentinel_revisit_map(
    download_dir: str,
    country_name: str,
    stac_items: pystac.ItemCollection,
    max_expected_revisit: int = 21,
    save_to: str = None,
):
    """Using a list of stac items, rasterize them to the country grid and use the difference
    between the previous acquisition and the current acquisition to compute a per-pixel
    histogram of revisit times.
    """

    base_dir = os.path.join(download_dir, country_name)
    country_geom = load_poly_from_geojson(
        os.path.join(base_dir, f"{country_name}.geojson")
    )

    base_grid = xr.open_dataarray(
        os.path.join(base_dir, f"{country_name}_grid.tif"), engine="rasterio"
    )

    revisit_histogram = np.zeros(
        (max_expected_revisit,) + tuple(base_grid.rio.shape), dtype=float
    )
    prev_acq_map = np.zeros(base_grid.rio.shape, dtype="datetime64[ns]")

    for item in reversed(list(stac_items)):
        current_acq = np.datetime64(
            datetime.strptime(
                item.properties["start_datetime"], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        )

        mask = rasterize(
            [item.geometry],
            base_grid.rio.shape,
            transform=base_grid.rio.transform(),
            all_touched=True,
        )

        idx = (current_acq - prev_acq_map).astype("timedelta64[D]")
        idx = idx / np.timedelta64(1, "D")  # Convert to an integer

        for i in range(max_expected_revisit):
            update_mask = (idx == i) & (mask > 0)
            if update_mask.any():
                revisit_histogram[i, update_mask] += 1

        prev_acq_map[mask > 0] = current_acq

    revisit_histogram_da = xr.DataArray(
        revisit_histogram,
        dims=("days", "y", "x"),
        coords={
            "days": range(1, max_expected_revisit + 1),
            "y": base_grid.y,
            "x": base_grid.x,
        },
    )
    revisit_histogram_da.attrs = base_grid.attrs
    revisit_histogram_da.rio.write_crs("EPSG:4326", inplace=True)
    revisit_histogram_da.rio.write_nodata(
        -sys.maxsize - 1,
        encoded=True,
        inplace=True,
    )

    revisit_histogram_da = revisit_histogram_da.rio.clip(
        [country_geom], all_touched=True, from_disk=True
    )

    if save_to:
        revisit_histogram_da.rio.to_raster(save_to)
        path, ext = os.path.splitext(save_to)
        np.save(f"{path}_prev_acq.npy", prev_acq_map, allow_pickle=True)

    return revisit_histogram_da


def get_average_revisit(revisit_histogram: xr.DataArray):
    product = revisit_histogram * revisit_histogram.days
    sum_product = product.sum(dim="days")
    total_count = revisit_histogram.sum(dim="days")
    return sum_product / total_count


def get_sentinel1_acquisitions_meta(
    download_dir: str,
    country_name: str,
    start_date: datetime,
    days_before: int = 30,
):
    format_date_range = (
        lambda sd, ed: f"{sd.strftime('%Y-%m-%d')}/{ed.strftime('%Y-%m-%d')}"
    )

    base_dir = os.path.join(download_dir, country_name)
    country_geom = load_poly_from_geojson(
        os.path.join(base_dir, f"{country_name}.geojson")
    )

    client = Client.open("https://earth-search.aws.element84.com/v1/")

    if days_before > 0:
        end_date = start_date - timedelta(
            days=1
        )  # As the STAC API will take this up until midnight if it is an end date
        start_date = end_date - timedelta(days=days_before)
    else:
        end_date = start_date - timedelta(days=days_before)

    search = client.search(
        collections=["sentinel-1-grd"],
        intersects=country_geom,
        datetime=format_date_range(start_date, end_date),
    )

    return list(search.items())


def read_ephem(path: str):
    dateparser = lambda x: pd.to_datetime(x, format="UTC=%Y-%m-%dT%H:%M:%S.%f")
    df = pd.read_xml(path, xpath=".//OSV")
    df.drop(columns=["TAI", "UT1"], inplace=True)
    df["UTC"] = df["UTC"].apply(dateparser)
    return df
