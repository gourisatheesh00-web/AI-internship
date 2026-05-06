#mallcustomer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("K-Means Clustering App")

# Load dataset
df = pd.read_csv("Mall_Customers[1].csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Slider for clusters
k = st.slider("Select number of clusters", 2, 10, 5)

# Train model
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot
fig, ax = plt.subplots()
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           s=200, c='black')

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation")

st.pyplot(fig)