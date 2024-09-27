import pandas as pd
import dataframe_preprocesser as dfp


def data_process(folder_path, scaler):

    weather_df = pd.read_csv(folder_path, parse_dates=["Idopont"])

    weather_df, last_date = dfp.process_df(weather_df)

    #the model only uses the last 700 datapoints to predict from
    weather_df = weather_df.iloc[-700:, :]

    scaled_df = scaler.fit_transform(weather_df)

    reverse_scale_help = pd.DataFrame(scaled_df).iloc[-300:, :].copy()

    return scaled_df, last_date, scaler, reverse_scale_help
