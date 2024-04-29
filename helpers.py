import os
import json
import requests
import sys
import rasterio
import rasterio.io

import pandas as pd
import numpy as np
import rioxarray as rxr
import xarray as xr

from contextlib import contextmanager
from rasterio.enums import Resampling
from typing import Iterable, List, Union, Tuple
from rasterio import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from shapely.geometry.base import BaseGeometry
from rasterio.features import rasterize
from shapely.geometry import shape, Polygon, box
from shapely.ops import transform
from pyproj import Transformer
from affine import Affine


def load_flood_events(download_dir: str, country: str, subset: str = None):
    flood_events = pd.read_csv(
        os.path.join(download_dir, country, "flood_events.csv"),
        parse_dates=["start_date", "end_date"],
    )

    if subset == "train":
        return flood_events.loc[flood_events.start_date < "2017-08-10"]
    elif subset == "test":
        return flood_events.loc[flood_events.start_date >= "2017-08-10"]


def load_poly_from_geojson(
    geojson_path: str,
) -> Polygon:
    geojson = json.load(open(geojson_path))
    bounds = shape(geojson["geometry"])
    return bounds


def download_file(url: str, destination: str):
    response = requests.get(url)
    with open(destination, "wb") as f:
        f.write(response.content)


def mask_raster_with_geometry(raster, transform, shapes, **kwargs):
    """Allow masking a numpy array using Rasterio in-memory rasters"""
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=raster.shape[1],
            width=raster.shape[2],
            count=1,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            dataset.write(raster)
            output, output_trans = rasterio.mask.mask(dataset, shapes, **kwargs)

    return output, output_trans


def bbox_to_transform(bbox: Union[BaseGeometry, tuple], xres=10, yres=10):
    if isinstance(bbox, BaseGeometry):
        bbox = bbox.bounds

    minx, miny, maxx, maxy = bbox
    affine = Affine(xres, 0, minx, 0, -yres, maxy)

    return affine


def bbox_to_grid(bbox: BaseGeometry, xres=10, yres=10):
    minx, miny, maxx, maxy = bbox.bounds

    minx = np.floor(minx * xres) / xres
    miny = np.floor(miny * yres) / yres
    maxx = np.floor(maxx * xres) / xres
    maxy = np.floor(maxy * yres) / yres
    clean_bbox = (minx, miny, maxx, maxy)

    width = int(np.ceil((maxx - minx) / xres))
    height = int(np.ceil((maxy - miny) / yres))

    transform = bbox_to_transform(clean_bbox, xres=xres, yres=yres)

    return transform, width, height


def reproject_geometry(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


def create_region_mask_and_grid(roi, resolution=1000, output_path="blank_raster.tif"):
    """Creates a raster which will be the base grid for our data stack."""
    roi_3857 = reproject_geometry(roi, "EPSG:4326", "EPSG:3857")

    transform, width, height = bbox_to_grid(
        box(*roi_3857.bounds),
        resolution,
        resolution,
    )

    crs_4326 = rasterio.crs.CRS.from_epsg(4326)
    crs_3857 = rasterio.crs.CRS.from_epsg(3857)
    transform, width, height = calculate_default_transform(
        crs_3857, crs_4326, width, height, *roi_3857.bounds
    )

    rasterized = rasterize([roi], out_shape=(height, width), transform=transform)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(rasterized, 1)

    return transform, width, height


@contextmanager
def open_on_grid(
    src_path: str,
    grid_path: str,
    resampling: Resampling = Resampling.nearest,
    crop_to: BaseGeometry = None,
    save_to: str = None,
    **kwargs,
):
    base_grid = xr.open_dataarray(grid_path, engine="rasterio")
    src = xr.open_dataarray(src_path, engine="rasterio", chunks={})

    # Force nodata as the GF DB rasters don't have it
    src = src.where(src != -sys.maxsize - 1, np.nan)
    src.rio.write_nodata(np.nan, encoded=True, inplace=True)

    src_reproj = src.rio.reproject_match(
        base_grid,
        resampling=resampling,
        nodata=np.nan,
    )

    if not crop_to is None:
        src_reproj = src_reproj.rio.clip(
            [crop_to],
            all_touched=True,
            from_disk=True,
            **kwargs,
        )
        src_reproj.rio.write_nodata(np.nan, encoded=True, inplace=True)

    if save_to:
        src_reproj.rio.to_raster(save_to)

    yield src_reproj

    src.close()
    src_reproj.close()
    base_grid.close()
