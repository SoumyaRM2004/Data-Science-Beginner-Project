import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px


df = pd.read_csv('Crop Production data.csv')


print("First few rows of the dataset:")
print(df.head())

print("\nColumn names in the dataset:")
print(df.columns)

print("\nData types and missing values:")
print(df.info())


df = df.dropna()
df['Crop_Year'] = df['Crop_Year'].astype(int)  


print("\nSummary statistics:")
print(df.describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['Production'], bins=30, kde=True)
plt.title('Distribution of Crop Production')
plt.xlabel('Production')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Crop_Year', y='Production', hue='Crop')
plt.title('Crop Production Over the Years')
plt.xlabel('Year')
plt.ylabel('Production')
plt.legend(title='Crop', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

df['Yield'] = df['Production'] / df['Area'] 
# Plotting Yield by Crop
plt.figure(figsize=(12, 8))
sns.boxplot(x='Crop', y='Yield', data=df)
plt.xticks(rotation=90)
plt.title('Yield by Crop')
plt.xlabel('Crop')
plt.ylabel('Yield')
plt.show()

features = ['Area']  
target = 'Production'


X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Production')
plt.xlabel('Actual Production')
plt.ylabel('Predicted Production')
plt.show()

avg_production = df.groupby(['State_Name', 'Crop'])['Production'].mean().reset_index()

fig = px.bar(avg_production, x='State_Name', y='Production', color='Crop', title='Average Production by State and Crop')
fig.show()

 
total_production = df['Production'].sum()
average_yield = df['Yield'].mean()
print(f"Total Production: {total_production}")
print(f"Average Yield: {average_yield}")

results = {
    'Metric': ['Total Production', 'Average Yield'],
    'Value': [total_production, average_yield]
}
results_df = pd.DataFrame(results)
results_df.to_csv('Crop_Production_Analysis_Results.csv', index=False)
