import pandas as pd
df = pd.read_csv('predicted_results.csv')

sum = df['predicted_total'].sum()
original_total_medal = df['Total'].sum()
df['prob'] = df['predicted_total']/sum
df['pre_medal'] = df['prob']*original_total_medal
df.to_csv('predicted_results.csv',index=False)

