import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Amazon Sales data.csv')


print("First few rows of the dataset:")
print(df.head())

print("\nColumn names in the dataset:")
print(df.columns)


sales_column = 'Total Revenue'


df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month


month_wise_sales = df.groupby('Month')[sales_column].sum()

year_wise_sales = df.groupby('Year')[sales_column].sum()

year_month_wise_sales = df.groupby(['Year', 'Month'])[sales_column].sum().unstack()


plt.figure(figsize=(10, 6))
month_wise_sales.plot(kind='bar', color='skyblue')
plt.title('Month-wise Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()


plt.figure(figsize=(10, 6))
year_wise_sales.plot(kind='bar', color='orange')
plt.title('Year-wise Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(year_month_wise_sales, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('Yearly Month-wise Sales Trend')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total Revenue', y='Total Profit', data=df)
plt.title('Relationship between Total Revenue and Total Profit')
plt.xlabel('Total Revenue')
plt.ylabel('Total Profit')
plt.show()

total_sales = df[sales_column].sum()
total_quantity = df['Units Sold'].sum()
average_discount = df['Unit Price'].mean() - df['Unit Cost'].mean()

print(f"Total Sales: ${total_sales:.2f}")
print(f"Total Quantity Sold: {total_quantity}")
print(f"Average Discount: ${average_discount:.2f}")


results = {
    'Metric': ['Total Sales', 'Total Quantity Sold', 'Average Discount'],
    'Value': [total_sales, total_quantity, average_discount]
}
results_df = pd.DataFrame(results)
results_df.to_csv('Sales_Analysis_Results.csv', index=False)
