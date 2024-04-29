"""
Using data from:

Baugh, Calum; Colonese, Juan; D'Angelo, Claudia; Dottori, Francesco; Neal, Jeffrey; 
Prudhomme, Christel; Salamon, Peter (2024): Global river flood hazard maps. 
European Commission, Joint Research Centre (JRC) [Dataset] 
PID: http://data.europa.eu/89h/jrc-floods-floodmapgl_rp50y-tif
"""

import os
from typing import List
import requests
import rasterio

import numpy as np
import xarray as xr
import rioxarray as rxr

from rasterio.merge import merge
from shapely.geometry import mapping
from tqdm import tqdm
from shapely.geometry import shape, Polygon, box

from helpers import load_poly_from_geojson, download_file, mask_raster_with_geometry

GLOFAS_BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
)
TILE_EXTENTS_URL = f"{GLOFAS_BASE_URL}/tile_extents.geojson"
FLOOD_EVENT_PERIODS = [10, 20, 50, 75, 100, 200, 500]


def get_intersecting_geometries(country_geom: Polygon):
    response = requests.get(TILE_EXTENTS_URL)
    tile_extents_geojson = response.json()

    intersecting_geometries = []
    country_bounds = box(*country_geom.bounds)

    for feature in tile_extents_geojson["features"]:
        tile_polygon = shape(feature["geometry"])
        if country_bounds.intersects(tile_polygon):
            intersecting_geometries.append(feature)

    return intersecting_geometries


VALID_FLOOD_EVENTS = [
    "RP10",
    "RP20",
    "RP50",
    "RP75",
    "RP100",
    "RP200",
    "RP500",
]


def download_flood_hazard_map_for_geometry(
    download_dir: str,
    country_geom: Polygon,
    flood_event_type: str = "RP100",
    progress_bar=tqdm,
):
    assert (
        flood_event_type in VALID_FLOOD_EVENTS
    ), f"flood_event_type must be one of {VALID_FLOOD_EVENTS}"

    tiles = get_intersecting_geometries(country_geom)
    merged_tif_path = os.path.join(download_dir, f"flood_hazard_{flood_event_type}.tif")

    tile_paths = []

    if os.path.exists(merged_tif_path):
        return merged_tif_path

    for tile in progress_bar(tiles, desc=f"Downloading Flood Hazard tiles"):
        tile_id = tile["properties"]["id"]
        tile_name = tile["properties"]["name"].upper()

        tile_url = f"{GLOFAS_BASE_URL}/{flood_event_type}/ID{tile_id}_{tile_name}_{flood_event_type}_depth.tif"
        tile_path = os.path.join(download_dir, f"ID{tile_id}_{tile_name}.tif")

        if not os.path.exists(tile_path):
            download_file(tile_url, tile_path)

        tile_paths.append(tile_path)

    src_files_to_mosaic = []
    for tile_path in tile_paths:
        src = rasterio.open(tile_path)
        src_files_to_mosaic.append(src)
        nodata = src.nodata

    mosaic, out_trans = merge(src_files_to_mosaic)

    clipped, clipped_trans = mask_raster_with_geometry(
        mosaic, out_trans, [country_geom], crop=True
    )

    # Write the mosaic to disk
    with rasterio.open(
        merged_tif_path,
        "w",
        driver="GTiff",
        crs=src.crs,
        transform=clipped_trans,
        width=clipped.shape[2],
        height=clipped.shape[1],
        count=1,
        dtype=clipped.dtype,
        nodata=nodata,
    ) as dest:
        dest.write(clipped)

    for tile_path in tile_paths:
        os.remove(tile_path)

    return merged_tif_path


def compute_flood_probability_map(
    download_dir: str,
    country_name: str,
    use_periods: List = FLOOD_EVENT_PERIODS,
):
    """Combine the hydrological model outputs into a flood probability map"
    base_dir = os.path.join(download_dir, country_name)

    country_geom = load_poly_from_geojson(
        os.path.join(base_dir, f"{country_name}.geojson")
    )

    prob = None
    total_prob = 0

    for event_period in use_periods:
        flood_hazard_path = os.path.join(
            base_dir, f"flood_hazard_RP{event_period}_grid.tif"
        )
        flood_hazard = xr.open_dataarray(flood_hazard_path, engine="rasterio")
        flood_hazard = flood_hazard.fillna(0)

        # Normalize the flood map from the hydrocological model as it can have outliers.
        flood_prob = flood_hazard.load().clip(max=flood_hazard.quantile(0.99))
        flood_prob = flood_prob / flood_prob.max()

        if prob is None:
            prob = (1 / event_period) * flood_prob
        else:
            prob = prob + ((1 / event_period) * flood_prob)

        total_prob += 1 / event_period

    normalized_prob = prob / total_prob
    normalized_prob = normalized_prob.rio.clip(
        [country_geom], all_touched=True, from_disk=True
    )

    return normalized_prob
