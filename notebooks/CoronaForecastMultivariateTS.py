import pandas as pd
from pathlib import Path
from loguru import logger
import shutil
import wandb
from flood_forecast.trainer import train_function
from corona_ts.core.config_train_main import generate_training_config
from corona_ts.data_utils.data_creator import loop_through_locations

wandb_sweep_config = {
    "name": "Multivariate TS sweep",
    "method": "grid",
    "parameters": {
        "batch_size": {
            "values": [2, 3, 4, 5]
        },
        "lr":  {
            "values": [0.001, 0.002, 0.004, 0.01]
        },
        "forecast_history":{
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        "out_seq_length":{
            "values": [1, 2, 3]
        }
    }
}

def format_corona_data(region_df:pd.DataFrame, region_name:str, training_data_directory=Path(".")):
    """
    Format data for a specific region into
    a format that can be used with flow forecast.
    """
    logger.info(f"currently runninng {region_df['region_identifier'].iloc[0]}")
    region_df = region_df.copy()
    region_df = region_df.sort_values(by="date").reset_index(drop=True)
    region_df.loc[:, 'datetime'] = region_df['date']

    region_df.loc[:, 'precip'] = 0
    region_df.loc[:, 'temp'] = 0
    region_df = region_df.fillna(0)
    region_df['new_cases'] = region_df['cases'].diff()
    region_df.loc[0, 'new_cases'] = 0
    region_df= region_df.fillna(method="backfill")
    region_df.to_csv(training_data_directory / f"{region_name}.csv", index=False)
    return region_df, len(region_df), training_data_directory / f"{region_name}.csv"

if __name__ == "__main__":
    data_dir = Path("../data")
    df = pd.read_csv(data_dir / "timeseries_with_weather_mobility.csv", parse_dates=["date"])
    logger.info("data read okay")
    
    locations_df = loop_through_locations(df)

    italian_regions = [
        'Italy__Calabria__',
        'Italy__Apulia__',
        'Italy__Basilicata__',
        'Italy__Emilia-Romagna__',
        'Italy__Sicily__',
        'Italy__Liguria__',
        'Italy__Tuscany__',
        'Italy__Marche__',
        'Italy__Lombardy__',
        'Italy__Abruzzo__',
        'Italy__Lazio__',
        'Italy__Sardinia__',
        'Italy__Veneto__',
        'Italy__Campania__',
        'Italy__Umbria__',
        'Italy__Molise__']

    logger.info("begin experiment")
    logger.info(f"regions of interest: {','.join(italian_regions)}")
    
    sweep_id = wandb.sweep(wandb_sweep_config, project="covid-forecast")
    temp_training_data_dir = data_dir / "temp"
    temp_training_data_dir.mkdir(exist_ok=True)

    query = "region_identifier=='{}'"
    for region in italian_regions:
        region_df, dataset_length, file_path = format_corona_data(df.query(query.format(region)), region, temp_training_data_dir)
        wandb.agent(
            sweep_id,
            lambda: train_function(
                "PyTorch", generate_training_config(str(file_path), feature_columns=[
                    'retail_recreation', 'grocery_pharmacy',
                   'parks', 'transit_stations', 'workplaces', 'residential',
                   'avg_temperature', 'min_temperature', 'max_temperature',
                   'relative_humidity', 'specific_humidity', 'pressure', "new_cases"],
                target_column=["new_cases"], df_len=dataset_length)))

    logger.info("done")
    shutil.rmtree(temp_training_data_dir)

