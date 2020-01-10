from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from plotly.offline import  plot

#Load Data
orders = pd.read_csv("olist_orders_dataset.csv") 
order_items=pd.read_csv("olist_order_items_dataset.csv")
customers=pd.read_csv("olist_customers_dataset.csv")
order_payments=pd.read_csv("olist_order_payments_dataset.csv")
products=pd.read_csv("olist_products_dataset.csv")


# New dataframe combines columns that we will be using and new columns we will store calculated data to
data= pd.merge(orders, customers[['customer_id', 'customer_unique_id']], on='customer_id', how = 'left')
data=pd.merge(data,order_items[['order_id', 'order_item_id','price']],on='order_id', how = 'left')
print(data.info())

data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])

data_recency = pd.DataFrame(data['customer_unique_id'].unique())
data_recency.columns = ['customer_unique_id']
#Time since last purchase(recency)
#Attach Last purchase column for each client 
clients_last_purchases = data.groupby('customer_unique_id').order_purchase_timestamp.max().reset_index()
#Create new label
clients_last_purchases.columns = ['customer_unique_id','last_purchase']

# Recency = number of days between last purchase and previous purchase
clients_last_purchases['recency'] = (clients_last_purchases['last_purchase'].max() - clients_last_purchases['last_purchase']).dt.days

# Merge new column to dataframe
data_recency = pd.merge(data_recency, clients_last_purchases[['customer_unique_id','recency']], on='customer_unique_id')
# Print data sample, print labels in dataframe
print(data_recency.head())
print(data_recency.recency.describe())
# Recency plot
fig = go.Figure(
    data=go.Histogram(x=data_recency['recency'],
        marker=dict(
        color = "lightgray",
        line = dict(color = "darkgray",
        width = 2)))
    )

fig.update_layout(
    title='Recency Distribution',
    yaxis= {'title': "Number of Customers"},
    xaxis= {'title': "Days"},
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro") 
)
fig.show()
#Elbow Method to find best cluster quantity
sum_squared_errors={}
recencyCount = data_recency[['recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(recencyCount)
    recencyCount["clusters"] = kmeans.labels_
    sum_squared_errors[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sum_squared_errors.keys()), list(sum_squared_errors.values()))
plt.xlabel("Clusters quantity")
plt.show()

# Unsupervised machine learning algorithm K-means for client segmentation (days between last purchase and previous purchase)
kmeans = KMeans(n_clusters=4)
kmeans.fit(data_recency[['recency']])
data_recency['recency_group'] = kmeans.predict(data_recency[['recency']])


#Ordering clusters
def order_cluster(group_label, label,dataframe,ascending):
    new_group_label = 'new_' + group_label
    new_dataframe = dataframe.groupby(group_label)[label].mean().reset_index()
    new_dataframe = new_dataframe.sort_values(by=label,ascending=ascending).reset_index(drop=True)
    new_dataframe['index'] = new_dataframe.index
    group_data = pd.merge(dataframe,new_dataframe[[group_label,'index']], on=group_label)
    group_data =group_data.drop([group_label],axis=1)
    group_data =group_data.rename(columns={"index":group_label})
    return group_data

data_recency = order_cluster('recency_group', 'recency',data_recency,False)
print(data_recency.head())
print(data_recency.recency.describe())

#Dataframe with total number of purchases 

data_frequency = data.groupby('customer_unique_id').order_purchase_timestamp.count().reset_index()
print(data_frequency.head())
data_frequency.columns = ['customer_unique_id','frequency']
print(data_frequency.frequency.describe())
data_recency = pd.merge(data_recency, data_frequency, on='customer_unique_id')
print(data_recency.head())

# Frequency Plot

fig = go.Figure(
    data=go.Histogram(x=data_recency.query('frequency < 1000')['frequency'],
        marker=dict(
        color = "lightgray",
        line = dict(color = "darkgray",
        width = 2)))
    )

fig.update_layout(
    title='Frequency Distribution',
    yaxis= {'title': "Number of Customers"},
    xaxis= {'title': "Number of Purchases"},
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro") 
)
fig.show()

#K-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(data_recency[['frequency']])
data_recency['frequencyCluster'] = kmeans.predict(data_recency[['frequency']])

#Purchases
data_recency = order_cluster('frequencyCluster', 'frequency',data_recency,True)
print(data_recency.groupby('frequencyCluster')['frequency'].describe())

#Calculate Monetary Value

data['monetaryValue'] = data['price'] * data['order_item_id']
data_monetaryValue = data.groupby('customer_unique_id').monetaryValue.sum().reset_index()

print(data_monetaryValue.monetaryValue.describe())


# Add Monetary Value column to main dataframe
data_recency = pd.merge(data_recency, data_monetaryValue, on='customer_unique_id')
print(data_recency.sample(5))
print(data_recency.info())

# Plot Monetary Value

fig = go.Figure(
    data=go.Histogram(x=data_recency.query("monetaryValue < 1000")['monetaryValue'],
        marker=dict(
        color = "lightgray",
        line = dict(color = "darkgray",
        width = 2)))
    )

fig.update_layout(
    title='Monetary Value',
    yaxis= {'title': "Number of Customers"},
    xaxis= {'title': "Spend per Customer"},
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro") 
)
fig.show()

# Clustering Monetary Value
kmeans = KMeans(n_clusters=4)
kmeans.fit(data_recency[['monetaryValue']])
data_recency['monetaryValueCluster'] = kmeans.predict(data_recency[['monetaryValue']])

data_recency = order_cluster('monetaryValueCluster', 'monetaryValue',data_recency,True)
print(data_recency.groupby('monetaryValueCluster')['monetaryValue'].describe())

#Overall Score
data_recency['total'] = data_recency['recency_group'] + data_recency['frequencyCluster'] + data_recency['monetaryValueCluster']
data_recency.groupby('total')['recency','frequency','monetaryValue'].mean()

data_recency['unit'] = 'low worth'
data_recency.loc[data_recency['total']>2,'unit'] = 'mid worth' 
data_recency.loc[data_recency['total']>4,'unit'] = 'high worth' 

# Plot segmentation 
plot = data_recency.query("monetaryValue < 50000 and frequency < 2000")

# Plot Frequency - Monetary Value

fig = go.Figure()

fig.add_scatter(x=plot.query("unit == 'low worth'")['frequency'],y=plot.query("unit == 'low worth'")['monetaryValue'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#925C16',
            opacity= 0.5), name='Low Worth')
fig.add_scatter(x=plot.query("unit == 'mid worth'")['frequency'],y=plot.query("unit == 'mid worth'")['monetaryValue'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#9A8811',
            opacity= 0.5), name='Medium Worth')
fig.add_scatter(x=plot.query("unit == 'high worth'")['frequency'],y=plot.query("unit == 'high worth'")['monetaryValue'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#BF6C1F',
            opacity= 0.5), name='High Worth')

fig.update_layout(
    yaxis= {'title': "Monetary Value"},
    xaxis= {'title': "Frequency"},
    title='Clusters',
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro")
    )
fig.show()

# Plot Recency - Monetary Value

fig = go.Figure()

fig.add_scatter(x=plot.query("unit == 'low worth'")['recency'],y=plot.query("unit == 'low worth'")['monetaryValue'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#925C16',
            opacity= 0.5), name='Low Worth')
fig.add_scatter(x=plot.query("unit == 'mid worth'")['recency'],y=plot.query("unit == 'mid worth'")['monetaryValue'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#9A8811',
            opacity= 0.5), name='Medium Worth')
fig.add_scatter(x=plot.query("unit == 'high worth'")['recency'],y=plot.query("unit == 'high worth'")['monetaryValue'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#BF6C1F',
            opacity= 0.5), name='High Worth')

fig.update_layout(
    yaxis= {'title': "Monetary Value"},
    xaxis= {'title': "Recency"},
    title='Clusters',
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro")
    )
fig.show()

# Plot Frequency - Recency


fig = go.Figure()

fig.add_scatter(x=plot.query("unit == 'low worth'")['recency'],y=plot.query("unit == 'low worth'")['frequency'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#925C16',
            opacity= 0.5), name='Low Worth')
fig.add_scatter(x=plot.query("unit == 'mid worth'")['recency'],y=plot.query("unit == 'mid worth'")['frequency'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#9A8811',
            opacity= 0.5), name='Medium Worth')
fig.add_scatter(x=plot.query("unit == 'high worth'")['recency'],y=plot.query("unit == 'high worth'")['frequency'],
        mode='markers',marker= dict(size= 9,
            line= dict(width=1),
            color= '#BF6C1F',
            opacity= 0.5), name='High Worth')

fig.update_layout(
    yaxis= {'title': "Recency"},
    xaxis= {'title': "Frequency"},
    title='Clusters',
    paper_bgcolor="#a67c17",
    plot_bgcolor="Gainsboro",
    font=dict(
        family='Courier New, monospace',
        size=14,
        color="Gainsboro"),
    titlefont=dict(
        family='Courier New, monospace',
        size=18,
        color="Gainsboro")
    )
fig.show()
