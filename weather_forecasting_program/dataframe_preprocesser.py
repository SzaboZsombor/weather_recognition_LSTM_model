import pandas as pd
import numpy as np


def columns_to_num(weather_df):

    # Celsius degree
    weather_df["Homérséklet"] = weather_df["Homérséklet"].str.replace(r'[^\d.]', '', regex=True)
    weather_df["Homérséklet"] = pd.to_numeric(weather_df["Homérséklet"])

    # Percentage
    weather_df["Páratartalom"] = weather_df["Páratartalom"].str.replace(r'[^\d.]', '', regex=True)
    weather_df["Páratartalom"] = pd.to_numeric(weather_df["Páratartalom"]).astype('float64')

    # hPa
    weather_df["Légnyomás"] = weather_df["Légnyomás"].str.replace(r'[^\d.]', '', regex=True)
    weather_df["Légnyomás"] = pd.to_numeric(weather_df["Légnyomás"])

    return weather_df


def process_date(weather_df):
    weather_df["day"] = weather_df["Idopont"].dt.day
    weather_df["hour"] = weather_df["Idopont"].dt.hour
    weather_df["month"] = weather_df["Idopont"].dt.month
    weather_df["minute"] = weather_df["Idopont"].dt.minute

    # encoding the periodical nature of time
    weather_df["minute_sin"] = np.sin(2 * np.pi * weather_df["minute"] / 59)
    weather_df["month_cos"] = np.cos(2 * np.pi * weather_df["minute"] / 59)

    weather_df["hour_sin"] = np.sin(2 * np.pi * weather_df["hour"] / 23)
    weather_df["hour_cos"] = np.cos(2 * np.pi * weather_df["hour"] / 23)

    weather_df["day_sin"] = np.sin(2 * np.pi * weather_df["day"] / 30)
    weather_df["day_cos"] = np.cos(2 * np.pi * weather_df["day"] / 30)

    weather_df["month_sin"] = np.sin(2 * np.pi * weather_df["month"] / 12)
    weather_df["month_cos"] = np.cos(2 * np.pi * weather_df["month"] / 12)

    last_date = weather_df.iloc[-1]["Idopont"]

    weather_df.drop(["Idopont", "month", "day", "hour", "minute"], axis=1, inplace=True)

    return weather_df, last_date


def process_df(weather_df):
    weather_df.dropna(how='all', inplace=True)

    weather_df.reset_index(drop=True, inplace=True)

    weather_df.drop(["Hoérzet", "Radiáció", "Csapadék 24h", "Szélsebesség", "Széllökés", "Szélirány",
                     "UV sugárzás", "Napsugárzás", "Unnamed: 13", "Harmatpont"], axis=1, inplace=True)

    weather_df = columns_to_num(weather_df)

    weather_df, last_date = process_date(weather_df)

    return weather_df, last_date
