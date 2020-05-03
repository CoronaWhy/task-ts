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
    time_series_path = DATA_DIR / "timeseries.csv"
    if not time_series_path.exists():
        logger.info("Time series not present locally, downloading...")
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
    # drop growth_factor due to 100% missing
    df = df.reindex(
        columns=[
            "country",
            "region",
            "sub_region",
            "city",
            "lat",
            "long",
            "date",
            "level",
            "tz",
            "cases",
            "deaths",
            "recovered",
            "active",
            "tested",
            "hospitalized",
            "discharged",
        ]
    )
    df["level"] = df["level"].map({"state": "region", "county": "sub_region"})

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


def load_data() -> pd.DataFrame:
    """Load time series data enriched with mobility data

    Returns:
        pd.DataFrame:
    """
    ts_df = time_series_formatter(fetch_time_series())
    mobility_df = fetch_mobility_data()

    metrics = [
        "retail_recreation",
        "grocery_pharmacy",
        "parks",
        "transit_stations",
        "workplaces",
        "residential",
    ]
    # Incorporate mobility data
    # Country level
    mobility_country_df = mobility_df.loc[
        mobility_df["region"].isnull(), metrics + ["country", "date"]
    ]
    mobility_country_df.columns = ["country_" + x for x in metrics] + [
        "country",
        "date",
    ]
    # Region level
    mobility_region_df = mobility_df.loc[
        ~mobility_df["region"].isnull(),
        metrics + ["country", "region", "date"],
    ]
    mobility_region_df.columns = ["region_" + x for x in metrics] + [
        "country",
        "region",
        "date",
    ]

    # Merge dataframe
    enriched_ts_df = pd.merge(
        ts_df, mobility_country_df, on=["country", "date"], how="left"
    )
    enriched_ts_df = pd.merge(
        enriched_ts_df,
        mobility_region_df,
        on=["country", "region", "date"],
        how="left",
    )
    return enriched_ts_df


if __name__ == "__main__":
    # for testing only
    df = load_data()
    df.head()
