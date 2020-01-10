# Using unsupervised machine learning algorithm K-means for RFM :shipit:

Context

The [Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) has information about 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers.

This [Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without the hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. 

After a customer purchases the product from Olist Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.

The module [k_means.py](https://github.com/icodeitnl//Unsupervised-Machine-Learning-algorithm-K-Means-for-RFM/blob/masterr/k_means.py) contains the scripts that bring the order and structure to the selected data.


**RFM** method analyses customer value. The abbreviation stands for the attributes used in segmentation, namely *recency, frequency, and monetary value*. **Frequency** determines how often the purchase is made, **recency** defines the most recent purchase and, finally, **monetary value** measures spend per customer.

In marketing terms, **client segmentation** splits business clients into groups that have common attributes based on behavioural, demographic, psychographic or geographic data. Customer segmentation enables companies to target specific groups, allowing effective allocation of resources, appropriate pricing, service, and product customisation,  strategizing and innovation.


To validate the number of clusters, **Elbow Method** is being used. It estimates the optimal value K produced by the cost function. While iterating through increasing *K values*, average distortion decreases and vice a versa. The “elbow” calculates the point where distortion declines or in other words if the plot looks like an arm, the elbow is where the forearm begins.

The *frequency* score table shows that most loyal customer made 24 purchases in 2 years period, while 84151 customers made 1 purchase and 10742 made 2 purchases.
