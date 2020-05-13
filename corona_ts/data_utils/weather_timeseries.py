from itertools import chain
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

from task_geo.dataset_builders.nasa import nasa
from data_crawler import load_data, DATA_DIR


def get_weather_time_series_for_one_location(df) -> pd.DataFrame:
    """
    Args:
        df: pandas.DataFrame with columns including ['country', 'region', 'sub_region', 'lon', 'lat'] (to fit
        with task_geo library)-- here, supply df for 1 unique lon/lat only to make it more efficient to parallelize

    Returns: original dataframe joined with weather data for 1 location

    """
    weather_columns = ['avg_temperature', 'min_temperature',
        'max_temperature', 'relative_humidity', 'specific_humidity',
        'pressure']
    weather = nasa(df, start_date=df.date.min(), end_date=df.date.max(), join=False)
    if weather is None:
        return pd.DataFrame(columns=list(df.columns) + weather_columns)
    weather = weather[[
        'date', 'avg_temperature', 'min_temperature',
        'max_temperature', 'relative_humidity', 'specific_humidity',
        'pressure']]
    df = df.set_index('date').merge(weather.set_index('date'), left_index=True, right_index=True)
    return df.reset_index()


def get_weather_time_series_all(df, parallel_jobs=20) -> pd.DataFrame:
    """

    Args:
        df: pd.DataFrame with ['country', 'region', 'sub_region', 'lon', 'lat'] for multiple locations
        parallel_jobs: number of parallel api calls to do at once.20 seems optimal in
        balancing time and not getting blocked by making too many calls at once

    Returns: original dataframe joined with weather data

    """
    # the api only has resolution of 0.5 degrees lat/lng, and you might get blocked if you call locations
    # that are too close together. This is an attempt to reduce the chances of calling very close locations at once
    unqiue_lat_long = df[['lat', 'long']].drop_duplicates().sample(frac=1.0)
    unique_dataframes = [df[(df.lat == lat) & (df.long==long)].rename(columns={'long': 'lon'}) for idx,( lat, long) in unqiue_lat_long.iterrows()]
    weather_results = Parallel(n_jobs=parallel_jobs, prefer='threads')(delayed(get_weather_time_series_for_one_location)(df) for df in tqdm(unique_dataframes))
    return pd.concat([x for x in weather_results if x is not None])


if __name__ == "__main__":
    # note: you might not want to keep redoing this as calling the data for all ~3000 locations we have
    # on the timeseries takes ~30mins on a good network connection
    filepath = DATA_DIR / 'timeseries_with_weather_mobility.csv'
    download = False
    if filepath.is_file() and not download:
        weather_df = pd.read_csv(filepath)
    else:
        df = load_data()
        weather_df = get_weather_time_series_all(df)
        weather_df.to_csv(DATA_DIR / 'timeseries_with_weather_mobility.csv', index=False)
