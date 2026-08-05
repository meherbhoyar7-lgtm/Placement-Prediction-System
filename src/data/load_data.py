import pandas as pd
import pandas as pd

def load_data():
    df = pd.read_csv("/Users/meherbhoyar/Desktop/Placement Prediction System/PlacementpredictionSystem/Data/placement_data.csv")
    return df
def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target":"placementStatus"
    }
if __name__ == "__main__":
    df=load_data()
    print(get_summary(df))