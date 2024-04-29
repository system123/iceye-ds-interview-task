import os
import requests
import pycountry
import json
import zipfile
import sys
import rasterio
import warnings

import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd

from rasterio.enums import Resampling
from tqdm import tqdm
from datetime import datetime
from tqdm import tqdm
from typing import List

from helpers import download_file, load_poly_from_geojson, load_flood_events
from sentinel1 import EPOCH_SENTINEL1

COUNTRY_GEOJSON_BASE_URL = "https://raw.githubusercontent.com/LonnyGomes/CountryGeoJSONCollection/master/geojson/"
GFD_BASE_URL = "http://global-flood-database.cloudtostreet.info"


def fetch_flood_events(country_code: str):
    url = os.path.join(GFD_BASE_URL, "collection", country_code)
    response = requests.get(url)
    events = json.loads(response.text)
    events = [os.path.basename(e) for e in events]
    return events


def filter_events(events: List[str], cutoff_date: datetime = EPOCH_SENTINEL1):
    """Filter the flood events to only those which occured after a specific date"""
    filtered_events = []
    for event in events:
        start_date_str = event.split("_From_")[1].split("_to_")[0]
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        if start_date > cutoff_date:
            filtered_events.append(event)
    return filtered_events


def download_files(
    download_path: str,
    events: List[str],
    progress_bar=tqdm,
):
    downloaded_files = []

    os.makedirs(download_path, exist_ok=True)
    os.makedirs(os.path.join(download_path, "precip"), exist_ok=True)
    os.makedirs(os.path.join(download_path, "flood"), exist_ok=True)

    for event in progress_bar(
        events,
        total=len(events),
        desc=f"Downloading {os.path.basename(download_path)}",
    ):
        event_csv_name = event.replace("From", "Precip_PERSIANN_from")

        csv_url = f"https://storage.googleapis.com/gfd_v1_precip/daily_PERSIANN_csv/{event_csv_name}.csv"
        zip_url = f"https://storage.googleapis.com/gfd_v1_4/{event}.zip"

        csv_dest = os.path.join(download_path, "precip", f"{event}.csv")
        zip_dest = os.path.join(download_path, "flood", f"{event}.zip")
        download_file(csv_url, csv_dest)
        download_file(zip_url, zip_dest)

        # Extract the .tif file from the downloaded ZIP file
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            # There is only one .tif file in each archive
            tif_file = [f for f in zip_ref.namelist() if f.endswith(".tif")][0]
            zip_ref.extract(tif_file, os.path.join(download_path, "flood"))
            tif_dest = os.path.join(download_path, "flood", tif_file)

        os.remove(zip_dest)

        event_details = event.split("_")
        downloaded_files.append(
            {
                "event_id": event_details[1],
                "start_date": datetime.strptime(event_details[3], "%Y%m%d"),
                "end_date": datetime.strptime(event_details[5], "%Y%m%d"),
                "flood": tif_dest,
                "precip": csv_dest,
            }
        )

    df = pd.DataFrame.from_records(downloaded_files)
    df.to_csv(os.path.join(download_path, "flood_events.csv"), index=False)

    return df


def download_data_for(
    download_dir: str,
    country_code: str = None,
    country_name: str = None,
    download_region_bounds: bool = True,
    download_all: bool = False,
    progress_bar=tqdm,
):
    if country_code is None and country_name is None:
        raise ValueError("Either `country_name` or `country_code` should be specified")

    if country_name:
        country = pycountry.countries.search_fuzzy(country_name)

        if len(country) == 0:
            raise Exception(f"No country was found for search {country_name}")

        country_code = country[0].alpha_3
    else:
        country = pycountry.countries.get(alpha_3=country_code)
        country_name = country.common_name

    download_path = os.path.join(download_dir, country_name)
    events = fetch_flood_events(country_code)
    filtered_events = filter_events(events, cutoff_date=EPOCH_SENTINEL1)

    if len(filtered_events) == 0:
        raise Exception(
            f"No flood events were found for [{country_code}] in the period [{EPOCH_SENTINEL1}]"
        )

    if os.path.exists(os.path.join(download_path, "flood_events.csv")):
        event_files = pd.read_csv(os.path.join(download_path, "flood_events.csv"))
    else:
        files_to_download = events if download_all else filtered_events
        event_files = download_files(
            download_path,
            files_to_download,
            progress_bar=progress_bar,
        )

    region_file_path = os.path.join(download_path, f"{country_name}.geojson")
    if not os.path.exists(region_file_path) and download_region_bounds:
        download_file(
            os.path.join(COUNTRY_GEOJSON_BASE_URL, f"{country_code}.geojson"),
            region_file_path,
        )

    return event_files


def accumulate_data(
    download_dir: str,
    country_name: str,
    progress_bar=tqdm,
    save_to_disk=True,
    subset=None,
):
    base_dir = os.path.join(download_dir, country_name)

    suffix = "" if subset is None else f"_{subset}"

    output_path = os.path.join(base_dir, f"flood_stack{suffix}.tif")
    if os.path.exists(output_path):
        return xr.open_dataarray(output_path)

    flood_events = load_flood_events(download_dir, country_name, subset=subset)
    country_geom = load_poly_from_geojson(
        os.path.join(base_dir, f"{country_name}.geojson")
    )

    base_grid = xr.open_dataset(
        os.path.join(base_dir, f"{country_name}_grid.tif"), engine="rasterio"
    )

    size = list(base_grid.band_data.shape)
    accumulated_data = np.zeros((3,) + tuple(size[1:]), dtype=np.float32)
    last_event_ids = np.zeros(size[1:], dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, row in progress_bar(flood_events.iterrows(), total=len(flood_events)):
            with xr.open_dataset(row["flood"], engine="rasterio", chunks={}) as src:
                src_flood_duration = src.band_data.sel(band=[1, 2, 3])

                # Keep track of how many times this area was part of a flooded region
                src_flood_duration.loc[dict(band=3)] = (
                    src_flood_duration.loc[dict(band=3)] >= 0
                )

                src_flood_duration = src_flood_duration.where(
                    src_flood_duration != -sys.maxsize - 1, np.nan
                )
                src_flood_duration.rio.write_nodata(np.nan, encoded=True, inplace=True)

                src_reproj = src_flood_duration.rio.reproject_match(
                    base_grid,
                    resampling=Resampling.max,
                    nodata=np.nan,
                )
                src_cropped = src_reproj.rio.clip(
                    [country_geom], all_touched=True, from_disk=True
                )
                src_cropped.rio.write_nodata(np.nan, encoded=True, inplace=True)

                event_ids = np.where(
                    np.isfinite(src_cropped.loc[dict(band=3)]),
                    row["event_id"],
                    0,
                )
                last_event_ids = np.maximum(last_event_ids, event_ids)

                accumulated_data += src_cropped.where(np.isfinite(src_cropped), 0)

    stacked_data = np.vstack((accumulated_data, last_event_ids[np.newaxis, :, :]))
    stack = xr.DataArray(
        stacked_data,
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3, 4], "y": base_grid.y, "x": base_grid.x},
    )

    stack.attrs = base_grid.band_data.attrs
    stack.attrs["band_names"] = [
        "flooded_count",
        "total_duration",
        "event_count",
        "last_event_id",
    ]

    if save_to_disk:
        stack = stack.where(stack != -sys.maxsize - 1, np.nan)
        stack.rio.write_nodata(np.nan, encoded=True, inplace=True)
        stack.rio.to_raster(output_path)

    return stack
