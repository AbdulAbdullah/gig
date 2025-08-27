
"""
Fuel Use Data Analysis Script
Date: August 13, 2025

This script performs a comprehensive analysis of vehicle fuel usage, cost, and mileage data.
It calculates fuel efficiency, cost trends, identifies outliers, and generates visualizations and summary statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filepath):
	"""Load and validate the fuel use dataset."""
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"Data file not found: {filepath}")
	# Read CSV, skip comment lines, assign column names
	df = pd.read_csv(filepath, comment='#', names=['Mileage', 'Cost', 'Fuel_Litres', 'Date'])
	# Drop rows with missing essential values
	df = df.dropna(subset=['Mileage', 'Fuel_Litres', 'Cost'])
	# Ensure numeric types
	df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
	df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
	df['Fuel_Litres'] = pd.to_numeric(df['Fuel_Litres'], errors='coerce')
	df = df.dropna(subset=['Mileage', 'Cost', 'Fuel_Litres'])
	df = df.reset_index(drop=True)
	return df

def compute_metrics(df):
	"""Compute derived metrics: miles driven, fuel efficiency, cost per litre."""
	df['Prev_Mileage'] = df['Mileage'].shift(1)
	df['Miles_Driven'] = df['Mileage'] - df['Prev_Mileage']
	# Avoid division by zero
	df['Miles_per_Litre'] = df['Miles_Driven'] / df['Fuel_Litres']
	df['Cost_per_Litre'] = df['Cost'] / df['Fuel_Litres']
	return df

def plot_metric(x, y, xlabel, ylabel, title, filename, color='blue'):
	"""Plot and save a line chart for a given metric."""
	plt.figure(figsize=(10, 5))
	plt.plot(x, y, marker='o', color=color)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()

def save_summary(df, columns, filename):
	"""Save descriptive statistics for selected columns."""
	summary = df[columns].describe()
	summary.to_csv(filename)

def detect_outliers(df, column, lower_quantile=0.05, upper_quantile=0.95):
	"""Identify outliers based on quantile thresholds."""
	lower = df[column].quantile(lower_quantile)
	upper = df[column].quantile(upper_quantile)
	outliers = df[(df[column] < lower) | (df[column] > upper)]
	return outliers

def main():
	# File paths
	data_file = 'CE2NMP_ResitData_FuelUse.csv'
	efficiency_plot = 'fuel_efficiency.png'
	cost_plot = 'cost_per_litre.png'
	summary_file = 'analysis_summary.csv'
	outliers_file = 'efficiency_outliers.csv'

	print('Loading data...')
	df = load_data(data_file)
	print(f'Data loaded: {len(df)} records')

	print('Computing metrics...')
	df = compute_metrics(df)

	print('Generating visualizations...')
	plot_metric(
		x=df['Mileage'],
		y=df['Miles_per_Litre'],
		xlabel='Mileage',
		ylabel='Miles per Litre',
		title='Fuel Efficiency (Miles per Litre)',
		filename=efficiency_plot
	)
	plot_metric(
		x=df['Mileage'],
		y=df['Cost_per_Litre'],
		xlabel='Mileage',
		ylabel='Cost per Litre',
		title='Cost per Litre Over Time',
		filename=cost_plot,
		color='orange'
	)

	print('Saving summary statistics...')
	save_summary(df, ['Miles_per_Litre', 'Cost_per_Litre'], summary_file)

	print('Detecting outliers...')
	outliers = detect_outliers(df, 'Miles_per_Litre')
	outliers.to_csv(outliers_file, index=False)

	print('Analysis complete. Charts and summary files generated.')

if __name__ == '__main__':
	main()
