#Bar graph for Renewable Internal Freshwater Resources per Capita (cubic meters)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for the time period of interest (1990 to 2022)
water_resource_data = processed_data[(processed_data['Country Name'].isin(country_of_interest)) & 
                                     (processed_data['Indicator Name'] == 'Renewable internal freshwater resources per capita (cubic meters)')]

# Extract relevant columns for plotting
water_resource_data = water_resource_data[['Country Name'] + list(map(str, range(1990, 2022, 5)))]

# Set 'Country Name' as the index for plotting
water_resource_data.set_index('Country Name', inplace=True)

# Convert data to numeric (in case it's not)
water_resource_data = water_resource_data.apply(pd.to_numeric, errors='coerce')

# Plot the data
plt.figure(figsize=(12, 6))
water_resource_data.transpose().plot(kind='bar', width=0.8)
plt.title('Renewable Internal Freshwater Resources per Capita (1990-2022) in 5-Year Intervals')
plt.xlabel('Year')
plt.ylabel('Renewable Internal Freshwater Resources per Capita (cubic meters)')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.tight_layout()
plt.show()









#Line graph for Population Growth (annual %)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for the time period of interest (1990 to 2022) and relevant indicator
population_growth_data = processed_data[(processed_data['Indicator Name'] == 'Population growth (annual %)') & 
                                        (processed_data['Country Name'].isin(country_of_interest))]

# Extract relevant columns for plotting
population_growth_data = population_growth_data[['Country Name'] + list(map(str, range(1990, 2023, 5)))]

# Set 'Country Name' as the index for plotting
population_growth_data.set_index('Country Name', inplace=True)

# Convert data to numeric (in case it's not)
population_growth_data = population_growth_data.apply(pd.to_numeric, errors='coerce')

# Plot the data
plt.figure(figsize=(12, 6))
population_growth_data.transpose().plot(kind='line', marker='o', linestyle='-')
plt.title('Population Growth (1990-2022) in 5-Year Intervals')
plt.xlabel('Year')
plt.ylabel('Population Growth (annual %)')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.tight_layout()
plt.show()








#Correlation Heatmap for Brazil: Water Scarcity and Agricultural Productivity

import seaborn as sns
import matplotlib.pyplot as plt

# Relevant indicators for water scarcity and agricultural productivity in Brazil
selected_columns_brazil = ['Population growth (annual %)', 
                            'Renewable internal freshwater resources per capita (cubic meters)',
                            'Annual freshwater withdrawals, total (billion cubic meters)',
                            'People using at least basic drinking water services (% of population)',
                            'Agriculture, forestry, and fishing, value added (% of GDP)']

# Filter the dataset for Brazil and relevant indicators
brazil_data = processed_data[(processed_data['Country Name'] == 'Brazil') & 
                              (processed_data['Indicator Name'].isin(selected_columns_brazil))]

# Extract relevant columns for correlation analysis
correlation_data = brazil_data.set_index('Indicator Name').loc[:, list(map(str, range(1990, 2022, 5)))]

# Convert data to numeric (in case it's not)
correlation_data = correlation_data.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
correlation_matrix = correlation_data.transpose().corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Brazil: Water Scarcity and Agricultural Productivity')
plt.show()





#Comparison of Arable Land and Forest Area (% of Land Area) - Bar chart

import matplotlib.pyplot as plt

# Update the list of countries of interest
country_of_interest = ['Malaysia', 'Qatar', 'India', 'China', 'Brazil', 'United Kingdom', 'Bangladesh']

# Relevant indicators for comparison
indicators_to_compare = ['Arable land (% of land area)', 'Forest area (% of land area)']

# Filter the dataset for the selected indicators and relevant countries
comparison_data = processed_data[(processed_data['Country Name'].isin(country_of_interest)) & 
                                 (processed_data['Indicator Name'].isin(indicators_to_compare))]

# Extract the years from the columns (excluding 2022)
years = comparison_data.columns[4:-1].astype(int).tolist()  # Convert to integers
years = [str(year) for year in years if year <= 2020]  # Exclude 2022

# Convert the selected columns to numeric
comparison_data[years] = comparison_data[years].apply(pd.to_numeric, errors='coerce')

# Pivot the data for better visualization
comparison_pivot = comparison_data.pivot_table(values=years, 
                                                index=['Country Name', 'Indicator Name'], 
                                                aggfunc='mean')

# Reset the index for plotting
comparison_pivot.reset_index(inplace=True)

# Plot the combined data for each indicator
plt.figure(figsize=(12, 6))
for indicator in indicators_to_compare:
    data = comparison_pivot[comparison_pivot['Indicator Name'] == indicator]
    plt.bar(data['Country Name'], data[years].mean(axis=1), label=indicator)

plt.title('Comparison of Arable Land and Forest Area (% of Land Area) (1990-2020 in 5-Year Intervals)')
plt.xlabel('Country')
plt.ylabel('Percentage')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()








#Public-Private Partnerships Investment in Water and Sanitation (1998-2022) - Pie chart

import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Define the selected countries
selected_countries = ['India', 'China', 'Brazil', 'Bangladesh']

# Filter the dataset for the selected countries and indicator
ppp_investment_data = processed_data[
    (processed_data['Country Name'].isin(selected_countries)) & 
    (processed_data['Indicator Name'] == 'Public private partnerships investment in water and sanitation (current US$)')]

# Extract relevant columns for plotting
ppp_investment_data = ppp_investment_data[['Country Name'] + list(map(str, range(1998, 2023, 5)))]

# Set 'Country Name' as the index for plotting
ppp_investment_data.set_index('Country Name', inplace=True)

# Convert data to numeric (in case it's not)
ppp_investment_data = ppp_investment_data.apply(pd.to_numeric, errors='coerce')

# Calculate the total investment for each country
total_investment = ppp_investment_data.sum(axis=1)

# Plotting a pie chart with more spacing for labels
plt.figure(figsize=(10, 10))
plt.pie(total_investment, labels=None, autopct='%1.1f%%', startangle=90, labeldistance=1.2)
plt.title('Public-Private Partnerships Investment in Water and Sanitation (1998-2022) - Selected Countries')

# Add legend with more space
plt.legend(total_investment.index, loc='upper right', bbox_to_anchor=(1.2, 1))

plt.show()










#Agricultural Methane and Nitrous Oxide Emissions (1990-2005) China and India - Bar graph

import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Select countries of interest
countries_of_interest = ['China', 'India']

# Filter the dataset for the time period of interest (1990 to 2010) and relevant indicators
emissions_data = processed_data[
    (processed_data['Indicator Name'].isin(['Agricultural methane emissions (% of total)', 'Agricultural nitrous oxide emissions (% of total)'])) & 
    (processed_data['Country Name'].isin(countries_of_interest))
]

# Extract relevant columns for plotting excluding the year 2010
emissions_data = emissions_data[['Country Name', 'Indicator Name'] + list(map(str, range(1990, 2010, 5)))].melt(
    id_vars=['Country Name', 'Indicator Name'], var_name='Year', value_name='Percentage'
)

# Set 'Country Name' and 'Year' as the index for plotting
emissions_data.set_index(['Country Name', 'Year', 'Indicator Name'], inplace=True)

# Convert data to numeric (in case it's not)
emissions_data['Percentage'] = pd.to_numeric(emissions_data['Percentage'], errors='coerce')

# Plot the data
plt.figure(figsize=(12, 6))
emissions_data.unstack('Indicator Name').plot(kind='bar', width=0.8)
plt.title('Agricultural Methane and Nitrous Oxide Emissions (1990-2005) in 5-Year Intervals for China and India')
plt.xlabel('Country, Year')
plt.ylabel('Percentage of Total Emissions')
plt.xticks(rotation=45)
plt.legend(title='Emission Type')
plt.tight_layout()
plt.show()











# Correlation Heatmap for India: Methane and Nitrous Oxide Emissions.

import seaborn as sns
import matplotlib.pyplot as plt

# Relevant indicators for water scarcity and agricultural productivity
selected_columns_india = [
    'Population growth (annual %)', 
    'Renewable internal freshwater resources per capita (cubic meters)',
    'Annual freshwater withdrawals, total (billion cubic meters)',
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'Agricultural methane emissions (% of total)',
    'Agricultural nitrous oxide emissions (% of total)'
]

# Filter the dataset for India and relevant indicators
india_data = processed_data[(processed_data['Country Name'] == 'India') & 
                             (processed_data['Indicator Name'].isin(selected_columns_india))]

# Extract relevant columns for correlation analysis
correlation_data = india_data.set_index('Indicator Name').loc[:, list(map(str, range(1990, 2022, 5)))]

# Convert data to numeric (in case it's not)
correlation_data = correlation_data.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
correlation_matrix = correlation_data.transpose().corr()

# Plot the correlation heatmap with a different color palette
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for India: Methane and Nitrous Oxide Emissions.')
plt.show()








# Correlation Heatmap for Malaysia: Agricultural land and the Population.

import seaborn as sns
import matplotlib.pyplot as plt

# Relevant indicators for Correlation Heatmap for Malaysia: Agricultural land and the Population.
selected_columns_malaysia = [
    'Population growth (annual %)', 
    'Arable land (% of land area)',
    'Population density (people per sq. km of land area)',
    'Fertilizer consumption (kilograms per hectare of arable land)',
    'Permanent cropland (% of land area)'
]

# Filter the dataset for Malaysia and relevant indicators
malaysia_data = processed_data[(processed_data['Country Name'] == 'Malaysia') & 
                                (processed_data['Indicator Name'].isin(selected_columns_malaysia))]

# Extract relevant columns for correlation analysis
correlation_data_malaysia = malaysia_data.set_index('Indicator Name').loc[:, list(map(str, range(1990, 2022, 5)))]

# Convert data to numeric (in case it's not)
correlation_data_malaysia = correlation_data_malaysia.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
correlation_matrix_malaysia = correlation_data_malaysia.transpose().corr()

# Plot the correlation heatmap with a different color palette ('rocket')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_malaysia, annot=True, cmap='rocket', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Malaysia: Agricultural land and the Population.')
plt.show()
