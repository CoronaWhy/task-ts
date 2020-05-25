import wandb
from datetime import datetime, date


def generate_training_config(file_path, feature_columns, target_column, df_len, weight_path=None, custom_config={}, wandb_run_name=None):
    run = wandb.init(project="covid-forecast", name=wandb_run_name)
    wandb_config = wandb.config
    train_number = df_len * .7
    validation_number = df_len * .9
    config_default = {
        "model_name": "MultiAttnHeadSimple",
        "model_type": "PyTorch",
        "model_params": {
            "number_time_series": len(feature_columns), #number of inputs to ts model, assume to be feature cols + the target column
            "seq_len": wandb_config["forecast_history"],
            "output_seq_len": wandb_config["out_seq_length"],
            "forecast_length": wandb_config["out_seq_length"]
        },
        "weight_path_add": {
            "excluded_layers": ["last_layer.weight", "last_layer.bias"]
        },
        "dataset_params":
            {"class": "default",
             "training_path": file_path,
             "validation_path": file_path,
             "test_path": file_path,
             "batch_size": wandb_config["batch_size"],
             "forecast_history": wandb_config["forecast_history"],
             "forecast_length": wandb_config["out_seq_length"],
             "train_end": int(train_number),
             "valid_start": int(train_number + 1),
             "valid_end": int(validation_number),
             # "target_col": ["new_cases"],
             "target_col": target_column,
             "relevant_cols": feature_columns,
             # "relevant_cols": ["new_cases", "month", "weekday"],
             "scaler": "StandardScaler",
             "interpolate": False
             },
        "training_params":
            {
                "criterion": "MSE",
                "optimizer": "Adam",
                "optim_params":
                    {

                    },
                "lr": wandb_config["lr"],
                "epochs": 10,
                "batch_size": wandb_config["batch_size"]

            },
        "GCS": False,

        "sweep": True,
        "wandb": False,
        "forward_params": {},
        "metrics": ["MSE"],
        "inference_params":
            {
                "datetime_start": "2020-4-20",
                "hours_to_forecast": 10,
                "test_csv_path": file_path,
                "decoder_params": {
                    "decoder_function": "simple_decode",
                    "unsqueeze_dim": 1
                },
                "dataset_params": {
                    "file_path": file_path,
                    "forecast_history": wandb_config["forecast_history"],
                    "forecast_length": wandb_config["out_seq_length"],
                    "relevant_cols": feature_columns,
                    "target_col": target_column,
                    "scaling": "StandardScaler",
                    "interpolate_param": False
                }
            }
    }
    if weight_path:
        config_default["weight_path"] = weight_path
    config = {**config_default, **custom_config}
    wandb.config.update(config_default)
    return config


def make_config_file(file_path, df_len, weight_path=None):
    run = wandb.init(project="covid-forecast")
    wandb_config = wandb.config
    train_number = df_len * .7
    validation_number = df_len * .9
    config_default = {
        "model_name": "MultiAttnHeadSimple",
        "model_type": "PyTorch",
        "model_params": {
            "number_time_series": 3,
            "seq_len": wandb_config["forecast_history"],
            "output_seq_len": wandb_config["out_seq_length"],
            "forecast_length": wandb_config["out_seq_length"]
        },
        "dataset_params":
            {"class": "default",
             "training_path": file_path,
             "validation_path": file_path,
             "test_path": file_path,
             "batch_size": wandb_config["batch_size"],
             "forecast_history": wandb_config["forecast_history"],
             "forecast_length": wandb_config["out_seq_length"],
             "train_end": int(train_number),
             "valid_start": int(train_number + 1),
             "valid_end": int(validation_number),
             "target_col": ["new_cases"],
             "relevant_cols": ["new_cases", "month", "weekday"],
             "scaler": "StandardScaler",
             "interpolate": False
             },
        "training_params":
            {
                "criterion": "MSE",
                "optimizer": "Adam",
                "optim_params":
                    {

                    },
                "lr": wandb_config["lr"],
                "epochs": 10,
                "batch_size": wandb_config["batch_size"]

            },
        "GCS": False,

        "sweep": True,
        "wandb": False,
        "forward_params": {},
        "metrics": ["MSE"],
        "inference_params":
            {
                "datetime_start": "2020-04-21",
                "hours_to_forecast": 10,
                "test_csv_path": file_path,
                "decoder_params": {
                    "decoder_function": "simple_decode",
                    "unsqueeze_dim": 1
                },
                "dataset_params": {
                    "file_path": file_path,
                    "forecast_history": wandb_config["forecast_history"],
                    "forecast_length": wandb_config["out_seq_length"],
                    "relevant_cols": ["new_cases", "month", "weekday"],
                    "target_col": ["new_cases"],
                    "scaling": "StandardScaler",
                    "interpolate_param": False
                }
            }
    }
    if weight_path:
        config_default["weight_path"] = weight_path
    wandb.config.update(config_default)
    return config_default


def wandb_make_config():
  sweep_config = {
    "name": "Default sweep",
    "method": "grid",
    "parameters": {
          "batch_size": {
              "values": [2, 3, 4, 5]
          },
          "lr":{
              "values":[0.001, 0.002, 0.004, 0.01]
          },
          "forecast_history":{
              "values":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          },
          "out_seq_length":{
              "values":[1, 2, 3]
          }
      }
  }
  return sweep_config
