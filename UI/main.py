import os
import sys
import pandas as pd

class App:
    def __init__(self,data_file_path:str="data/my_data.csv"):
        self.__dlc_path = str()
        self.__csv_path = str()
        self.__output_df = pd.DataFrame()
        self.__data_file_path = self.__resource_path(data_file_path)

    def get_paths_from_user(self)->str:
        self.__dlc_path = input("Enter the path for DLC_color:\nFor example: C:\\Users\\username\\Desktop\\DLC_color\n")
        while(self.__csv_path.endswith('.csv') == False):
            self.__csv_path = input("Enter the path for csv result(including file itself):\nFor example: C:\\Users\\username\\Desktop\\DLC_color\\outputname.csv\n")
            if self.__csv_path.endswith('.csv') == False:
                print("Please enter a valid path for csv result")

    def __resource_path(self, relative_path:str)->str:
        """ Get the absolute path to the resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def do_something(self)->pd.DataFrame:
        dict = {
            'Column1': [1, 2, 3, 4, 5],
            'Column2': ['A', 'B', 'C', 'D', 'E']
        }
        self.__output_df = pd.DataFrame(dict)
        print(f"DataFrame create successfully")
    
    def save_df_to_csv(self)->None:
        self.__output_df.to_csv(self.__csv_path, index=False)
        print(f"DataFrame saved successfully to {self.__csv_path}")

def main():
    app = App("data/my_data.csv")

    try:
        app.get_paths_from_user()
        app.do_something()
        app.save_df_to_csv()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
