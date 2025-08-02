import fastf1
import os
import pandas as pd
import numpy as np

# load the last three years' race session data
Race_2024 = fastf1.get_session(2024, 'Hungarian Grand Prix', 'R')
Race_2023 = fastf1.get_session(2023, 'Hungarian Grand Prix', 'R')
Race_2022 = fastf1.get_session(2022, 'Hungarian Grand Prix', 'R')
Race_2022.load()
Race_2023.load()
Race_2024.load()

# calculate the position change during each year
Pos_2024 = Race_2024.results[["Abbreviation", "Position", "GridPosition"]].copy()
Pos_2024['Change_2024'] = Pos_2024['GridPosition'] - Pos_2024['Position']
Pos_2023 = Race_2023.results[["Abbreviation", "Position", "GridPosition"]].copy()
Pos_2023['Change_2023'] = Pos_2023['GridPosition'] - Pos_2023['Position']
Pos_2022 = Race_2022.results[["Abbreviation", "Position", "GridPosition"]].copy()
Pos_2022['Change_2022'] = Pos_2022['GridPosition'] - Pos_2022['Position']

# merging all the data into one dataframe
Pos_2024 = Pos_2024.merge(Pos_2023[['Abbreviation', 'Change_2023']], on="Abbreviation", how="left")
Pos_2024 = Pos_2024.merge(Pos_2022[['Abbreviation', 'Change_2022']], on="Abbreviation", how="left")

# '+' means positions gained & '-' means position lost
CngPos_2024 = Pos_2024.drop(['Position', 'GridPosition'],axis=1)

# NA values are replaced by the mean of the position change in other years.
CngPos_2024.iloc[:, 1:] = CngPos_2024.iloc[:, 1:].apply(lambda row: row.fillna(row.mean()), axis=1)

# calculate the mean for each row, excluding the first column
CngPos_2024['Avg_Change'] = CngPos_2024.iloc[:, 1:].mean(axis=1).round(2)
Avg_Change = CngPos_2024[['Abbreviation','Avg_Change']].copy()

Avg_Change.rename(columns={'Abbreviation': 'Driver'},inplace=True)
Avg_Change
