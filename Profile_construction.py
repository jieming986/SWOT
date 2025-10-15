import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pyproj
import re
from glob import glob
from shapely.geometry import Point
from pypinyin import lazy_pinyin
from multiprocessing.dummy import Pool as ThreadPool, Lock
from statsmodels.nonparametric.smoothers_lowess import lowess
from multiprocessing import Pool

def csv_to_shapefile(input_folder, output_folder):
    print("=== Step 1: CSV to Shapefile ===")

    os.makedirs(output_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_file_path = os.path.join(input_folder, csv_file)

        csv_data = pd.read_csv(csv_file_path)

        print(f"Processing file: {csv_file_path}")

        geometry = [Point(xy) for xy in zip(csv_data['longitude'], csv_data['latitude'])]

        gdf = gpd.GeoDataFrame(csv_data, geometry=geometry)

        gdf.crs = "EPSG:4326"

        shp_file_path = os.path.join(output_folder, f"{os.path.splitext(csv_file)[0]}.shp")

        gdf.to_file(shp_file_path)
        print(f"Shapefile successfully saved to {shp_file_path}")

    print("All CSV files have been converted to Shapefiles.")


def shapefile_projection_transform(input_folder, output_folder):
    print("=== Step 2: Shapefile Projection Transform ===")

    os.makedirs(output_folder, exist_ok=True)

    def get_utm_projection(zone):
        return pyproj.CRS(f"EPSG:326{zone:02d}")

    for filename in os.listdir(input_folder):
        if filename.endswith(".shp"):
            match = re.search(r'UTM(\d{2})[A-Z]*_', filename)
            if match:
                zone = int(match.group(1))
                print(f"Processing file: {filename} with UTM Zone: {zone}")

                target_crs = get_utm_projection(zone)

                input_shp = os.path.join(input_folder, filename)

                gdf = gpd.read_file(input_shp)

                print(f"Original CRS: {gdf.crs}")

                gdf = gdf.to_crs(target_crs)

                output_shp = os.path.join(output_folder, filename)

                gdf.to_file(output_shp)
                print(f"Projection complete for {filename}. Output saved to {output_shp}.")
            else:
                print(f"Failed to extract UTM zone from {filename}. Skipping.")


def swot_station_spatial_matching(station_shp_path, swot_shp_folder, output_folder):
    print("=== Step 3: SWOT Station Spatial Matching ===")

    station_gdf = gpd.read_file(station_shp_path)

    os.makedirs(output_folder, exist_ok=True)

    lock = Lock()

    def process_swot_file(swot_file):
        if not swot_file.endswith('.shp'):
            return

        swot_file_path = os.path.join(swot_shp_folder, swot_file)
        swot_gdf = gpd.read_file(swot_file_path)

        swot_gdf = swot_gdf[swot_gdf.geometry.type == 'Point']

        time_str = swot_file.split('_')[16]
        time_ymd = time_str[:8]
        time_hms = time_str[9:15]
        time = pd.to_datetime(f"{time_ymd} {time_hms}", format="%Y%m%d %H%M%S")

        time_decim = swot_gdf["time_decim"].iloc[0] if "time_decim" in swot_gdf.columns else np.nan

        for _, station_row in station_gdf.iterrows():
            station_name = station_row["renumber"]
            station_geom = station_row.geometry

            station_name_pinyin = ''.join(lazy_pinyin(str(station_name))).lower()

            disdam = station_row.get("disdam", np.nan)
            disdamkm = station_row.get("disdamkm", np.nan)
            renumber = station_row.get("renumber", np.nan)
            redisout = station_row.get("redisout", np.nan)

            points_in_station = swot_gdf[swot_gdf.within(station_geom)]

            if not points_in_station.empty:
                elevation_median = points_in_station["wse"].median()
                ct_median = points_in_station["cross_trac"].median()
                elevation_std = points_in_station["wse"].std()
                raster_num = len(points_in_station)

                output_data = {
                    "time": [time],
                    "time_ymd": [time.date()],
                    "time_decim": [time_decim],
                    "shuiwei_swot": [elevation_median],
                    "std_swot": [elevation_std],
                    "ct_median": [ct_median],
                    "raster_num": [raster_num],
                    "disdam": [disdam],
                    "disdamkm": [disdamkm],
                    "renumber": [renumber],
                    "redisout": [redisout],
                    "longitude": [station_row["x"]],
                    "latitude": [station_row["y"]],
                }
                output_df = pd.DataFrame(output_data)

                output_csv_path = os.path.join(output_folder, f"{station_name_pinyin}.csv")

                with lock:
                    if os.path.exists(output_csv_path):
                        output_df.to_csv(output_csv_path, mode='a', header=False, index=False)
                    else:
                        output_df.to_csv(output_csv_path, index=False)

                    try:
                        final_df = pd.read_csv(output_csv_path)
                        final_df["time"] = pd.to_datetime(final_df["time"])
                        final_df = final_df.sort_values(by="time")
                        final_df.to_csv(output_csv_path, index=False)
                    except pd.errors.EmptyDataError:
                        print(f"Warning: Failed to read {output_csv_path}, file is empty, skipping sort.")

    pool = ThreadPool(16)
    pool.map(process_swot_file, os.listdir(swot_shp_folder))
    pool.close()
    pool.join()

    print("Processing complete, results saved to output folder.")


def batch_rename_files(src_folder, dst_folder, prefix="316_"):
    print("=== Step 4: Batch Rename Files ===")

    os.makedirs(dst_folder, exist_ok=True)

    csv_files = glob(os.path.join(src_folder, "*.csv"))

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        new_file_name = f"{prefix}{file_name}"
        new_file_path = os.path.join(dst_folder, new_file_name)

        shutil.copy(file_path, new_file_path)
        print(f"Copied: {file_name} -> {new_file_name}")

    print("Batch rename and copy complete!")


def group_data_by_date(input_folder, output_folder):
    print("=== Step 5: Group Data by Date ===")

    os.makedirs(output_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    all_data = []

    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        if "time_ymd" in df.columns and "disdamkm" in df.columns:
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    grouped = df_all.groupby("time_ymd")

    for date, group in grouped:
        group = group.sort_values(by="disdamkm")

        output_file = os.path.join(output_folder, f"{date}.csv")

        group.to_csv(output_file, index=False)

    print("Data grouping by date complete, files saved!")


def process_file_lowess(args):
    fname, input_folder, output_folder, thresholds = args

    fpath = os.path.join(input_folder, fname)
    df = pd.read_csv(fpath)

    if {'disdamkm', 'shuiwei_swot'}.issubset(df.columns):
        df = df.dropna(subset=['disdamkm', 'shuiwei_swot'])

        df['filename'] = fname

        df_filtered = df.copy()

        for th in thresholds:
            frac = 0.2
            smoothed = lowess(df['shuiwei_swot'], df['disdamkm'], frac=frac, return_sorted=False)
            residuals = np.abs(df['shuiwei_swot'].values - smoothed)
            df_filtered = df[residuals <= th].copy()

        save_path = os.path.join(output_folder, fname)
        df_filtered.to_csv(save_path, index=False)

        return df, df_filtered
    else:
        return None, None


def plot_filtered_data(filtered_data, show_plot=True):
    plt.figure(figsize=(10, 6))
    for df_raw, df_filtered in filtered_data:
        if df_raw is not None and df_filtered is not None:
            plt.scatter(df_filtered['disdamkm'], df_filtered['shuiwei_swot'], s=5, alpha=0.6)

    plt.gca().invert_xaxis()
    plt.xlabel("disdamkm")
    plt.ylabel("shuiwei_swot")
    plt.title("Filtered Water Surface Elevation (LOWESS)")
    plt.ylim(148, 157)
    plt.grid(True)
    plt.tight_layout()

    if show_plot:
        plt.show()


def lowess_filtering(input_folder, output_folder, thresholds=None, show_plot=True):
    print("=== Step 6: LOWESS Filtering ===")
    os.makedirs(output_folder, exist_ok=True)

    if thresholds is None:
        thresholds = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    file_names = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    args_list = [(fname, input_folder, output_folder, thresholds) for fname in file_names]

    with Pool() as pool:
        filtered_data = pool.map(process_file_lowess, args_list)

    if show_plot:
        plot_filtered_data(filtered_data, show_plot)

    print("LOWESS filtering complete, results saved to output folder.")


if __name__ == '__main__':
    base_path = " " # Output file path
    input_folder_raw = " " # Iutput file path

    output_folder_stage1 = os.path.join(base_path, "1")
    output_folder_stage2 = os.path.join(base_path, "2")
    output_folder_stage3 = os.path.join(base_path, "3")
    output_folder_stage4 = os.path.join(base_path, "4")
    output_folder_stage5 = os.path.join(base_path, "5")
    output_folder_stage6 = os.path.join(base_path, "6")

    station_shp_path = r" " #SWORD

    csv_to_shapefile(input_folder_raw, output_folder_stage1)
    shapefile_projection_transform(output_folder_stage1, output_folder_stage2)
    swot_station_spatial_matching(station_shp_path, output_folder_stage2, output_folder_stage3)
    batch_rename_files(output_folder_stage3, output_folder_stage4)
    group_data_by_date(output_folder_stage4, output_folder_stage5)
    lowess_filtering(output_folder_stage5, output_folder_stage6)

    print("=== All steps complete ===")