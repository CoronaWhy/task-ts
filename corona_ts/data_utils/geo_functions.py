import pandas as pd

MOBILITY_REPORTS_URL = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'


def mobility_connector() -> pd.DataFrame:
    """Retrieves the Google mobility report."""
    return pd.read_csv(MOBILITY_REPORTS_URL)


def mobility_formatter(raw):
    # Put column names in lowercase alphanumerical
    column_names = {
        'country_region_code': 'country_iso',
        'country_region': 'country',
        'sub_region_1': 'region',
        'sub_region_2': 'sub_region',
        'date': 'date',
        'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
        'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy',
        'parks_percent_change_from_baseline': 'parks',
        'transit_stations_percent_change_from_baseline': 'transit_stations',
        'workplaces_percent_change_from_baseline': 'workplaces',
        'residential_percent_change_from_baseline': 'residential',
    }

    raw = raw.rename(columns=column_names)

    numeric_columns = [
        'retail_recreation', 'grocery_pharmacy', 'parks',
        'transit_stations', 'workplaces', 'residential'
    ]
    raw[numeric_columns] = raw[numeric_columns].astype(float)
    raw['date'] = pd.to_datetime(raw.date)
    column_order = [
        'country_iso', 'country', 'region', 'sub_region', 'date', 'retail_recreation',
        'grocery_pharmacy', 'parks', 'transit_stations', 'workplaces', 'residential'
    ]
    return raw[column_order]

def mobility():
    """Retrieve  the mobility reports from Google.
    Arguments:
        None
    Returns:
        pandas.DataFrame
    Example:
    >>> from task_geo.data_sources import get_data_source
    >>> mobility = get_data_source('mobility')
    """
    raw = mobility_connector()
    return mobility_formatter(raw)