import pandas as pd
import json
df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")
count = {}
for i in range(df.shape[0]):
    row = df.iloc[i]
    noc = row['NOC']
    year = row['Year']
    year = int(year)
    if noc in count:
        if year in count[noc]:
            count[noc][year] += 1
        else:
            count[noc][year] = 1
    else:
        count[noc] = {}
        count[noc][year] = 1

with open("athlete_count.json", "w", encoding="utf-8") as json_file:
    json.dump(count, json_file, indent=4)