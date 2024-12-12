import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import silhouette_score
import plotly.express as px
import numpy as np
from joypy import joyplot


file_path = 'OnlineRetail.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='ISO-8859-1') 


print('data before pp')
print(df.head())

print(df.isnull().sum())

print(df.info())


data = df.copy()

data = data.dropna(subset=['CustomerID'])

data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

print(data['InvoiceDate'].head())
#data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d-%m-%Y %H:%M')
data.loc[:, 'InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d-%m-%Y %H:%M')

data['TotalSales'] = data['Quantity'] * data['UnitPrice']

print('data after pp')
print(data.describe())


customer_data = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSales': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalSales': 'Monetary'
}).reset_index()

print('RFM analysis')
print(customer_data.head())


sns.histplot(customer_data['Monetary'], bins=50)
plt.title('Distribution of Monetary Values')
plt.show()

top_items = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_items.plot(kind='bar', title='Top 10 Items Sold')
plt.show()



# to detect outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

cleaned_data = remove_outliers(data, 'TotalSales')

country_data_cleaned = cleaned_data.groupby('Country').agg({
    'CustomerID': 'nunique',
    'TotalSales': 'sum',
    'InvoiceNo': 'count'
}).rename(columns={
    'CustomerID': 'UniqueCustomers',
    'InvoiceNo': 'TotalInvoices'
}).reset_index()

top_countries_cleaned = country_data_cleaned.nlargest(10, 'TotalSales')

plt.figure(figsize=(14, 8))


sns.violinplot(
    x='Country',
    y='TotalSales',
    data=cleaned_data[cleaned_data['Country'].isin(top_countries_cleaned['Country'])],
    density_norm='width',
    hue='Country',
    palette='coolwarm'
)

plt.xticks(rotation=45, fontsize=12, ha='right')
plt.xlabel('Country', fontsize=14, labelpad=10)

plt.ylabel('Total Sales (in $)', fontsize=14, labelpad=10)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.title('Violin Plot: Total Sales Distribution by Top 10 Countries',
          fontsize=16, fontweight='bold', pad=20)

plt.annotate(
    text="Total Sales Distribution by Top 10 Countries",
    xy=(0.95, 0.01), xycoords='axes fraction', ha='right', fontsize=10, color='gray'
)

plt.tight_layout()
plt.show()



cleaned_data['LogTotalSales'] = cleaned_data['TotalSales'].apply(lambda x: np.log1p(x))

filtered_data = cleaned_data[cleaned_data['Country'].isin(top_countries_cleaned['Country'])]

plt.figure(figsize=(14, 10))
joyplot(
    data=filtered_data,
    by='Country',
    column='LogTotalSales',
    figsize=(12, 8),
    colormap=plt.cm.viridis,
    linewidth=1.25,
    title="Ridgeline Plot: Log-Scaled Total Sales by Country"
)

plt.title('Ridgeline Plot of Total Sales by Top 10 Countries', fontsize=16, pad=20)
plt.xlabel('Log-Scaled Total Sales', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.tight_layout()
plt.show()



pivot_data_cleaned = top_countries_cleaned.pivot_table(index='Country', values='TotalInvoices')

plt.figure(figsize=(12, 8))



top_customers = data.groupby('CustomerID')['TotalSales'].sum().reset_index().nlargest(10, 'TotalSales')

top_customers['CustomerID'] = top_customers['CustomerID'].astype(str)

fig = px.bar(
    top_customers,
    x='CustomerID',
    y='TotalSales',
    text='TotalSales',
    title='Top 10 Customers by Total Sales',
    labels={'CustomerID': 'Customer ID', 'TotalSales': 'Total Sales ($)'},
    color='TotalSales',
    color_continuous_scale='Viridis'
)

fig.update_layout(
    xaxis=dict(title='Customer ID', tickmode='linear'),
    yaxis=dict(title='Total Sales ($)'),
    title_x=0.5, 
    coloraxis_showscale=True,
)

fig.update_traces(
    texttemplate='%{text}', 
    textposition='outside',
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1.5
)

fig.show()







model = IsolationForest(contamination=0.01)
customer_data['Outlier'] = model.fit_predict(customer_data[['Recency', 'Frequency', 'Monetary']])

print('number of outliers')
print(customer_data['Outlier'].value_counts())


scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Segment'] = kmeans.fit_predict(scaled_data)

sns.scatterplot(x='Recency', y='Monetary', hue='Segment', data=customer_data, palette='viridis')
plt.title('Customer Segments')
plt.show()

# data.to_csv('apriori.csv', index= False)


# #
# basket = data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
# basket = basket.applymap(lambda x: 1 if x > 0 else 0)
#
# frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
# rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
#
# print(rules.head())


silhouette = silhouette_score(scaled_data, customer_data['Segment'])
print('silhouette score')
print(f'Silhouette Score: {silhouette}')


fig = px.scatter(customer_data, x='Recency', y='Monetary', color='Segment', title='Customer Segments')
fig.show()



