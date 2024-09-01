import pandas as pd
import os
from utils.convert_raw_to_df import convert_raw_to_df
from utils.feature_calc import feature_calc
from utils.predictions import predict_and_plot_start, predict_and_plot_turn, plot_trajectory_to_file

class App:
    def __init__(self):
        self.__data_path = str()
        self.__csv_path = str()
        self.__output_df = pd.DataFrame()
        self.__first_pth = str()
        self.__second_pth = str()

    def get_paths_from_user(self)->str:
        data_path_valid = False
        while not data_path_valid:
            self.__data_path = input("Enter the full path for the Data folder:\nFor example: C:\\Users\\username\\Desktop\\Data\n")
            if not os.path.exists(self.__data_path):
                print("The provided path does not exist. Please enter a valid path.")
            else:
                print("The provided path is valid.")
                self.__csv_path = self.__data_path + "\\output.csv"
                data_path_valid = True
        while(self.__csv_path.endswith('.csv') == False):
            self.__first_pth = input("Enter the full path for first pth(including file itself):\nFor example: C:\\Users\\username\\Desktop\\Data\\start_move_model.pth\n")
            self.__second_pth = input("Enter the full path for second pth(including file itself):\nFor example: C:\\Users\\username\\Desktop\\Data\\stop_turn_model.pth\n")
            if self.__csv_path.endswith('.csv') == False:
                print("Error! Please enter a valid path for csv result, ending with .csv")

    def start_movement_and_turning_point_predictions(self)->pd.DataFrame:
        merged_df = convert_raw_to_df(self.__data_path)
        df_test_features = feature_calc(merged_df)

        start_models = 'start_move_model.pth'
        start_model_name = {'start_move_model.pth': 'start_move'}
        start_features = ['scaled_rotation_x', 'scaled_rotation_y','normalized_time','scaled_velocity','scaled_angle']
        start_model_path = f'C:\Users\avita\source\repos\ofekm5\ExtractingTurningPointGUI\UI\utils\{start_models}'
        start_df_predicted = predict_and_plot_start(df_test_features, start_models, start_model_name, start_features, start_model_path)

        turn_features = ['scaled_rotation_x', 'scaled_rotation_y','normalized_time','scaled_velocity','scaled_angle','scaled_angle_to_end','scaled_curvature','scaled_distance_to_end']
        turn_models = 'stop_turn_model.pth'
        turn_model_name = {'stop_turn_model.pth': 'stop_turn'}
        turn_model_path = f'C:\Users\avita\source\repos\ofekm5\ExtractingTurningPointGUI\UI\utils\{turn_models}'
        turn_df_predicted = predict_and_plot_turn(df_test_features, turn_models, turn_model_name, turn_features, turn_model_path)

        merged_prediction = start_df_predicted.merge(turn_df_predicted, on='id', how='outer')
        columns_to_drop = [col for col in merged_prediction.columns if col.endswith('_y')]
        merged_prediction.drop(columns=columns_to_drop, inplace=True)
        merged_prediction.columns = merged_prediction.columns.str.replace('_x', '', regex=False)

        plot_trajectory_to_file(merged_prediction, df_test_features, 'trajectories.png')

        self.__output_df = merged_prediction  
    
    def save_df_to_csv(self)->None:
        self.__output_df.to_csv(self.__csv_path, index=False)
        print(f"results saved successfully to {self.__csv_path}")


#/Users/avitalvasiliev/Documents/GitHub/ExtractingTurningPointGUI/t011222
#start_model_path = f'C:\Users\avita\source\repos\ofekm5\ExtractingTurningPointGUI\UI\utils\{start_models}'
def main():
    app = App()

    try:
        app.get_paths_from_user()
        app.start_movement_and_turning_point_predictions()
        app.save_df_to_csv()
        input("Press Enter to exit...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
