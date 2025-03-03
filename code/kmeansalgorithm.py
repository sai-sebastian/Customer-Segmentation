import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set random seed for reproducibility
np.random.seed(42)

# Read the CSV file containing distributor locations
file_path = 'DistributorData.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Ensure the CSV contains 'latitude' and 'longitude' columns
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    raise ValueError("CSV file must contain 'latitude' and 'longitude' columns.")

# Define the number of clusters (warehouses)
n_warehouses = 5  

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_warehouses, init='k-means++', max_iter=47, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# Extract final warehouse locations (centroids)
centroids = kmeans.cluster_centers_

# Create a DataFrame for centroids
centroid_df = pd.DataFrame(centroids, columns=['latitude', 'longitude'])
centroid_df['Warehouse'] = [f'Warehouse {i+1}' for i in range(n_warehouses)]

# Print the centroid locations
print(centroid_df)

# Visualization
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Plot each cluster
for i in range(n_warehouses):
    cluster_points = df[df['Cluster'] == i]
    plt.scatter(cluster_points['longitude'], cluster_points['latitude'], 
                label=f'Cluster {i+1}', alpha=0.6, color=colors[i])

# Plot centroids
plt.scatter(centroid_df['longitude'], centroid_df['latitude'], 
            color='black', marker='X', s=200, label='Warehouses')

# Annotate warehouse names
for i, warehouse in enumerate(centroid_df['Warehouse']):
    plt.annotate(warehouse, (centroid_df['longitude'][i], centroid_df['latitude'][i]), 
                 color='black', fontsize=12, ha='right')

plt.title('Optimal Warehouse Locations using K-Means')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
