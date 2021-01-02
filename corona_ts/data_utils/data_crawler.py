from typing import Dict, List
from pathlib import Path
from loguru import logger
import urllib.request

import pandas as pd
from task_geo.data_sources import get_data_source

#DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR = Path.cwd().parent / "data"


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
    time_series_path = DATA_DIR / "timeseries.csv"
    logger.info("Pulling current time series data, downloading...")
    url = "https://coronadatascraper.com/timeseries.csv"
    urllib.request.urlretrieve(url, time_series_path)

    time_series_df = pd.read_csv(time_series_path)
    return time_series_df


def time_series_formatter(df: pd.DataFrame) -> pd.DataFrame:
    """Format time series data

    Args:
        df (pd.DataFrame): Raw data from https://coronadatascraper.com/timeseries.csv

    Returns:
        pd.DataFrame
    """
    df = df.rename(
        columns={
            "growthFactor": "growth_factor",
            "state": "region",
            "county": "sub_region",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    # drop city level
    df = df.loc[~(df["level"] == "city")]
    # drop growth_factor due to 100% missing
    df = df.reindex(
        columns=[
            "country",
            "region",
            "sub_region",
            "lat",
            "long",
            "date",
            "level",
            "cases",
            "deaths",
            "recovered",
            "active",
            "tested",
            "hospitalized",
            "discharged",
        ]
    )
    df["level"] = df["level"].map(
        {"country": "country", "state": "region", "county": "sub_region"}
    )

    metrics = [
        "cases",
        "deaths",
        "recovered",
        "active",
        "tested",
        "hospitalized",
        "discharged",
    ]
    df[metrics] = df[metrics].fillna(0).astype(int)
    return df.sort_values(by="date").reset_index(drop=True)


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


def load_data() -> pd.DataFrame:
    """Load time series data enriched with mobility data

    Returns:
        pd.DataFrame:
    """
    index_cols = ["level", "country", "region", "sub_region", "date"]
    ts_df = time_series_formatter(fetch_time_series())
    ts_df = ts_df.set_index(index_cols)
    # drop duplicated rows
    ts_df = ts_df[~ts_df.index.duplicated()]

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


if __name__ == "__main__":
    # for testing only
    df = load_data()
    df.head().to_csv("test.csv")
