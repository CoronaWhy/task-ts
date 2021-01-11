from typing import Dict, List
from pathlib import Path
from loguru import logger
import urllib.request

import pandas as pd
from task_geo.data_sources import get_data_source

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def fetch_time_series() -> pd.DataFrame:
    """Fetch raw time series data from coronadatascraper.com
    Returns:
        pd.DataFrame: raw timeseries data at county/sub-region level
    """
    """ Old function
        if not time_series_path.exists():
            logger.info("Time series not present locally, downloading...")
            url = "https://coronadatascraper.com/timeseries.csv"
            urllib.request.urlretrieve(url, time_series_path)
    """
    time_series_path = "timeseries.csv"
    logger.info("Pulling current time series data, downloading...")
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    urllib.request.urlretrieve(url, "file.csv")
    df = pd.read_csv("file.csv")
    df = df.drop(columns=["iso2", "iso3", "code3", "FIPS"])
    df = df.melt(id_vars=["UID", "Admin2", "Province_State", "Country_Region", "Lat", "Long_", "Combined_Key"], var_name="date", value_name="cases")
    df = pd.read_csv("file.csv")
    df = df.drop(columns=["iso2", "iso3", "code3", "FIPS"])
    df = df.melt(id_vars=["UID", "Admin2", "Province_State", "Country_Region", "Lat", "Long_", "Combined_Key"], var_name="date", value_name="cases")
    df.columns = ["UID", "sub_region", "region", "country", "lat", "long", "Combined_Key", "date", "cases"]
    df["level"] = "sub_region"
    df["country"] = "United States"
    df["date"] = pd.to_datetime(df["date"])
    df["sub_region"] = df["sub_region"] + " County"
    return df



def load_df() -> pd.DataFrame:
    """Load time series data enriched with mobility data
    Returns:
        pd.DataFrame:
    """
    index_cols = ["level", "country", "region", "sub_region", "date"]
    #ts_df = time_series_formatter(fetch_time_series())
    #ts_df = ts_df.set_index(index_cols)
    # drop duplicated rows
    #ts_df = ts_df[~ts_df.index.duplicated()]
    ts_df = fetch_time_series()
    mobility_df = fetch_mobility_data()
    metrics = [
        "retail_recreation",
        "grocery_pharmacy",
        "parks",
        "transit_stations",
        "workplaces",
        "residential",
    ]

    mobility_df.loc[mobility_df["region"].isnull(), "level"] = "country"
    mobility_df.loc[
        (~mobility_df["region"].isnull())
        & (mobility_df["sub_region"].isnull()),
        "level",
    ] = "region"
    mobility_df.loc[
        ~mobility_df["sub_region"].isnull(), "level"
    ] = "sub_region"

    mobility_df = mobility_df.set_index(index_cols)[metrics]
    mobility_df.columns = ["mobility_" + x for x in metrics]

    # treat missing values in mobility data
    mobility_df = _treat_mobility_missing_values(mobility_df)

    # Incorporate mobility data
    enriched_ts_df = ts_df.reset_index().merge(mobility_df.reset_index(), how='inner',on=index_cols)
    return enriched_ts_df.reset_index()


def fetch_mobility_data() -> pd.DataFrame:
    mobility = get_data_source("mobility")
    return mobility()


def _find_min_values(df: pd.DataFrame) -> Dict[str, float]:
    """Return the min of each column as a dictionary

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        Dict[str, float]: dictionary with min valus for each columns
    """
    return df.min().to_dict()


def _treat_mobility_missing_values(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """Treat missing values in mobility data

    Due to privacy reasons, missing values in mobility data are present
    when there is not sufficient data to ensure anonymity.
    More details: 
    https://www.google.com/covid19/mobility/data_documentation.html?hl=en#about-this-data

    Missing values are due to lack of data, ie. lack of visits. 
    Therefore, we can infer that visits to a location dropped significantly.

    Treating missing value strategy:
    - Use the mimimum value of a (location, column) to fill missing value
    - When all data are missing for a (location, column), fillna with 0

    Args:
        mobility_df (pd.DataFrame): raw mobility dataframe

    Returns:
        pd.DataFrame: treated mobility dataframe
    """
    logger.info("Treat mobility missing values.")
    treated_dataframes: List[pd.DataFrame] = []
    levels = ["sub_region", "region", "country"]
    for i, level in enumerate(levels):
        df = mobility_df.loc[level]
        df = df.groupby(levels[i:]).apply(
            lambda x: x.fillna(_find_min_values(x))
        )
        df = df.fillna(0).reset_index()
        df["level"] = level
        treated_dataframes.append(df)

    treated_df = pd.concat(treated_dataframes)

    return treated_df.set_index(["level"] + levels[::-1] + ["date"])



if __name__ == "__main__":
    # for testing only
    df = load_df()
    df.head().to_csv("test.csv")
