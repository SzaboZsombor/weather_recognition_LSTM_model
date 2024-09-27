from datetime import timedelta
import numpy as np
import pandas as pd


def dt_to_date(last_date):

    temp_date = last_date

    pred_dates = []

    for i in range(0, 300):

        temp_date += timedelta(minutes=15)

        pred_dates.append(temp_date)

    return np.array(pred_dates)


def output_process(scaled_temp, scaler, last_date, reverse_scale_help):

    reverse_scale_help.iloc[:, 0] = scaled_temp



    re_scaled_df = pd.DataFrame(scaler.inverse_transform(reverse_scale_help))

    temp = re_scaled_df.iloc[:, 0]

    dates = dt_to_date(last_date)

    return np.array(temp), np.array(dates)
