import pandas as pd
import os

class App:
    def __init__(self,data_file_path:str="data/my_data.csv"):
        self.__data_path = str()
        self.__csv_path = str()
        self.__output_df = pd.DataFrame()

    def get_paths_from_user(self)->str:
        data_path_valid = False
        while not data_path_valid:
            self.__data_path = input("Enter the full path for the Data folder:\nFor example: C:\\Users\\username\\Desktop\\Data\n")
            if not os.path.exists(self.__data_path):
                print("The provided path does not exist. Please enter a valid path.")
            else:
                print("The provided path is valid.")
                data_path_valid = True
        while(self.__csv_path.endswith('.csv') == False):
            self.__csv_path = input("Enter the full path for csv result(including file itself):\nFor example: C:\\Users\\username\\Desktop\\Data\\outputname.csv\n")
            if self.__csv_path.endswith('.csv') == False:
                print("Error! Please enter a valid path for csv result, ending with .csv")

    def do_something(self)->pd.DataFrame:
        dict = {
            'Column1': [1, 2, 3, 4, 5],
            'Column2': ['A', 'B', 'C', 'D', 'E']
        }
        self.__output_df = pd.DataFrame(dict)
        print(f"results calculated successfully")
    
    def save_df_to_csv(self)->None:
        self.__output_df.to_csv(self.__csv_path, index=False)
        print(f"results saved successfully to {self.__csv_path}")

def main():
    app = App("data/my_data.csv")

    try:
        app.get_paths_from_user()
        app.do_something()
        app.save_df_to_csv()
        input("Press Enter to exit...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
