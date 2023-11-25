#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('FULL_B_COMBINED.csv')

#%%

new_data_list = []

categories = ["cities", "companies", "animals", "elements", "inventions", "facts"]
metrics = ["Acc on Pos Labels", "Acc on Neg Labels", "Unexpected_True / Size", "Unexpected_False / Size"]

# Group by 'System Prompt' and 'User Prompt' to ensure each unique combination gets its own row
for (system_prompt, user_prompt, prefix), group in df.groupby(['System Prompt', 'User Prompt', 'Prefix']):
    row_data = {
        'System Prompt': system_prompt,
        'User Prompt': user_prompt,
        'Prefix': prefix
    }
    
    unexpected_true_values = []
    unexpected_false_values = []
    
    for category in categories:
        for metric in metrics:
            column_name = f"{category}_{metric.replace(' ', '_').lower()}"
            value = group[group['Dataset'] == category][metric].values[0]
            
            # Save the values for later average calculation
            if metric == "Unexpected_True / Size":
                unexpected_true_values.append(value)
            elif metric == "Unexpected_False / Size":
                unexpected_false_values.append(value)
            else:
                if metric == "Acc on Pos Labels":
                    column_name=f"{category}Pos"
                elif metric == "Acc on Neg Labels":
                    column_name=f"{category}Neg"
                row_data[column_name] = value
    
    # Calculate the averages and add them to the row data
    row_data['avg_unexpected_true'] = sum(unexpected_true_values) / len(unexpected_true_values)
    row_data['avg_unexpected_false'] = sum(unexpected_false_values) / len(unexpected_false_values)
    
    new_data_list.append(row_data)

new_df = pd.DataFrame(new_data_list)

print(new_df)


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you've already loaded and transformed the dataframe as 'new_df'

# Drop the 'System Prompt' and 'User Prompt' columns
table_df = new_df.drop(columns=['System Prompt', 'User Prompt', 'Prefix'])

# Round off each number to the nearest whole number
table_df = (table_df * 100).astype(int)

# Create a table and remove axis
fig, ax = plt.subplots(figsize=(15, 35))  # set the size that you'd like (width, height)
ax.axis('off')
tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')

# Adjust the font size in the cells
tbl.auto_set_font_size(False)
tbl.set_fontsize(15)
tbl.scale(1.5, 3.5)  # Adjust the scale of the table cells

plt.show()


# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you've already loaded and transformed the dataframe as 'new_df'

# Drop the 'System Prompt' and 'User Prompt' columns
table_df = new_df.drop(columns=['System Prompt', 'User Prompt', 'Prefix'])

# Convert to percentages and round off to whole numbers
table_df = (table_df * 100).astype(int)

# Create a colormap to map values to colors
colormap = plt.cm.PiYG
norm = plt.Normalize(table_df.values.min()-25, table_df.values.max()+25)
colors = colormap(norm(table_df.values))

# Create a table and remove axis
fig, ax = plt.subplots(figsize=(15, 35))  # set the size that you'd like (width, height)
ax.axis('off')
tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center', cellColours=colors)

# Adjust the font size in the cells
tbl.auto_set_font_size(False)
tbl.set_fontsize(15)
tbl.scale(1.5, 3.5)  # Adjust the scale of the table cells

plt.show()

# %%

#%%
pd.set_option('display.max_colwidth', None)
print(new_df.iloc[:, :2])
# %%


