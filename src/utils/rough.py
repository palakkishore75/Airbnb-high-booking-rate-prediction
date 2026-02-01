import pandas as pd

df = pd.read_csv("/Users/percival/Projects/Airbnb_Analysis/outputs/submissions/high_booking_rate_group12.csv")



print(df.isna().sum())