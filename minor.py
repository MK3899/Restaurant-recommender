import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, atan2, sqrt
from sklearn.cluster import KMeans 
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

rating = pd.read_csv('RATINGS1.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')
dataset = pd.read_csv('Book1.csv')

#NEW USER
X = dataset.iloc[:,[1,2]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

n_clusters = 3

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 1, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 1, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 1, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 10, c = 'yellow', label = 'Centroids')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.show()

kmeans.cluster_centers_[:, 0]
kmeans.cluster_centers_[:, 1]

centroid = [kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1]]
Centroids = np.array(centroid)
Centroids

print("Input coordinates of your location:")
lat1 = radians(float(input("Latitude: ")))
lon1 = radians(float(input("Longitude: ")))

R = 6373.0   #Radius of Earth

distance = []
for i in range (n_clusters):
    lat2 = radians(Centroids[0][i])
    lon2 = radians(Centroids[1][i])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    dist = R * c
    #dist = 6371 * acos(sin(loc_lat)*sin(Centroids[0][i]) + cos(loc_lat)*cos(Centroids[0][i])*cos( Centroids[1][i]-loc_lon ))
    distance.append(dist)
print(distance)
    
min_dist = min(distance)
nearest_cluster_id = distance.index(min_dist)
print(nearest_cluster_id)

cluster = pd.DataFrame(list(zip(X[y_kmeans == nearest_cluster_id,0], X[y_kmeans == nearest_cluster_id,1])), columns =['latitude', 'longitude'])

rating_2 = rating.groupby('placeID').mean()
#rating_2['placeID'] = rating_2.index
rating_2.reset_index(inplace = True)

cuisine_2 = cuisine.groupby('placeID').agg(lambda col: ' '.join(col))
#cuisine_2['placeID'] = cuisine_2.index
cuisine_2.reset_index(inplace = True)

dataset_2 = pd.merge(dataset,cluster)
dataset_3 = pd.merge(dataset_2,rating_2)
dataset_4 = pd.merge(dataset_3,cuisine_2)

data = dataset_4
sub = input('Enter the cuisine:')
start = 0 
data["Indexes"]= data["Rcuisine"].str.find(sub, start) 
data 

subset_data = data[data['Indexes'] != -1]
subset_data

result = subset_data.sort_values('mean rating', ascending = False)
print(result)
    
similarity_dataset = rating.pivot(index = 'placeID', columns = 'userID', values='mean rating')
similarity_dataset = similarity_dataset.fillna(0)




def standardize(row):
    new_row = (row - row.mean())/(row.max() - row.min())
    return new_row
similarity_dataset = similarity_dataset.apply(standardize)
    
similarity_dataset.reset_index(inplace = True)


#NEW RESTAURANT
restaurants = pd.read_csv('Book1.csv')
restaurants = restaurants.sort_values('placeID', ascending = False)
print(restaurants)

Y = restaurants.iloc[:,[0,1,2]].values
restaurants_2 = pd.DataFrame(Y,columns= ['placeID','latitude','longitude'])


cuisine_3 = cuisine.groupby(['Rcuisine']).agg(lambda col: ' '.join(col))
cuisine_3.reset_index(inplace = True)
no_of_cuisines = cuisine_3.shape[0]
for i in range(1,60):
    cuisine_3.at[i-1,'placeID'] = i

cuisine_3.columns = ['Rcuisine','cuisine_number']

cuisine_4 = pd.merge(cuisine,cuisine_3)
restaurants_3 = pd.merge(restaurants_2,cuisine_4)
Z = restaurants_3.iloc[:,[1,2,4]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(Z)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++')
y_kmeans = kmeans.fit_predict(Z)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(Z[y_kmeans == 0,0], Z[y_kmeans == 0,1], Z[y_kmeans == 0,2], c = 'red')
ax.scatter(Z[y_kmeans == 1,0], Z[y_kmeans == 1,1], Z[y_kmeans == 1,2], c = 'blue')
ax.scatter(Z[y_kmeans == 2,0], Z[y_kmeans == 2,1], Z[y_kmeans == 2,2], c = 'green')
ax.scatter(Z[y_kmeans == 3,0], Z[y_kmeans == 3,1], Z[y_kmeans == 3,2], c = 'black')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c = 'yellow')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Cuisine')

plt.show()
