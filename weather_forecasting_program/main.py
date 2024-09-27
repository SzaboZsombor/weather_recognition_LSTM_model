import pandas as pd
import os
import model_predicting as mp
import data_preprocesser as dpr
import data_postprocesser as dpo
from sklearn.preprocessing import MinMaxScaler


def to_one_file(folder_path, file_path_df):  #this function take in the last days datas in different excel files and makes it into one file
    all_files = os.listdir(folder_path)

    df = pd.DataFrame()
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_excel(file_path, engine='xlrd', header=2, index_col=None, parse_dates=["Idopont"])
        df = pd.concat([df, temp_df], axis=0)

    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(file_path_df, index=False)


def save_predictions(temperature, dates):  #saves the predictions

    temp_dict = {"Date": dates, "Temperature": temperature}
    pred_df = pd.DataFrame(temp_dict)

    path = r"D:\Machine Learning\weather_prediction\predictions\prediction1.csv"

    pred_df.to_csv(path, index=False)


def predict(file_path):  #predicts the output values (also pre and post processes the data), the final output is the temperature at given dates

    scaler = MinMaxScaler()

    scaled_df, last_date, scaler, reverse_scale_help = dpr.data_process(file_path, scaler)

    pred_temp = mp.predict(scaled_df)
    pred_temp = pred_temp.flatten()


    temperature, dates = dpo.output_process(pred_temp, scaler, last_date, reverse_scale_help)

    return temperature, dates


def main():

    #this is the folder where all the excel files are which will make into one (last 8 days before prediction)
    folder_path = r"D:\Machine Learning\weather_prediction\2024_09_21_excel_files"

    #this is where the input csv file will be
    file_path_df = r"D:\Machine Learning\weather_prediction\after_2024_09_21\2024_09_21.csv"

    to_one_file(folder_path, file_path_df)

    temperature, dates = predict(file_path_df)

    save_predictions(temperature, dates)


if __name__ == "__main__":
    main()
