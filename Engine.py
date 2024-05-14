import pandas as pd
import matplotlib.pyplot as plt
import pylab
import scipy
from matplotlib import pyplot
from MLPipeline.Gaussian_Stationary import Gaussian_Stationary
from MLPipeline.Gaussian_Trend import Gaussian_Trend

# Importing the data from an Excel file
raw_csv_data = pd.read_excel("Input/CallCenterData.xlsx")

# Create a copy of the data for checkpoint
df_comp = raw_csv_data.copy()

# Convert the 'month' column to timestamps
df_comp["timestamp"] = df_comp["month"].apply(lambda x: x.timestamp())

# Set the 'month' column as the date index
df_comp.set_index("month", inplace=True)

# Set the frequency to monthly
df_comp = df_comp.asfreq('M')

# Plot the 'Healthcare' data
df_comp.Healthcare.plot(figsize=(20, 5), title="Healthcare")
plt.savefig("Output/" + "dataplot_healthcare.png")

# Check for normality:

# Density plot for 'Healthcare'
df_comp["Healthcare"].plot(kind='kde', figsize=(20, 10))
pyplot.savefig("Output/" + "Densityplot.png")

# QQ plot for 'Healthcare'
scipy.stats.probplot(df_comp["Healthcare"], plot=pylab)
plt.title("QQ plot for Healthcare")
pylab.savefig("Output/" + "QQPLot.png")

# Gaussian Processes:

# Create a new DataFrame with 'timestamp' and 'Healthcare'
data_df = df_comp[["timestamp", "Healthcare"]]

# Apply Gaussian Trend analysis
Gaussian_Trend(data_df)

# Calculate the first-order difference
df_comp["delta_1_Healthcare"] = df_comp.Healthcare.diff(1)

# Plot the first-order difference
df_comp.delta_1_Healthcare.plot(figsize=(20, 5))

# Checking the normality of the difference with Density Plots
df_comp["delta_1_Healthcare"].plot(kind='kde', figsize=(20, 10))
pyplot.savefig("Output/" + "difference.png")

# Create a new DataFrame with 'timestamp' and 'delta_1_Healthcare'
data_df_res = df_comp[["timestamp", "delta_1_Healthcare"]]

# Apply Gaussian Stationary analysis
Gaussian_Stationary(df_comp, data_df_res)
