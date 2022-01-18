import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 


def get_data():
	filename = "../karate1.embeddings"
	df = pd.read_csv(filename, sep=' ', header=None)
	node = df.iloc[:,0].to_numpy()
	dim = df.iloc[:,1:] # pandas dataframe
	std_dim = StandardScaler().fit_transform(dim) # numpy array
	# print(std_dim.describe())
	return [node, std_dim]

def do_pca(std_dim):
	pca = PCA(n_components=3)
	p_comp = pca.fit_transform(std_dim)

	# print(type(p_comp))
	# pfdf = pd.DataFrame(data=p_comp, columns=['pcp1', 'pcp2', 'pcp3'])
	# print(pfdf.head())

	# print(pca.explained_variance_ratio_)
	# print(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
	# print(sum(pca.explained_variance_ratio_))

	return p_comp

def print_pca_graph(data):
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	ax.scatter3D(data[:,0], data[:,1], data[:,2], color = "green")
	plt.title("PCA results")
	ax.set_xlabel("1st principal component")
	ax.set_ylabel("2nd principal component")
	ax.set_zlabel("3rd principal component")

	# show plot
	plt.show()

def set_color(label):
	if label == 0:
		return 'r'
	elif label == 1:
		return 'g'
	else:
		return 'b'

def print_kmeans_graph(result):
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	# for color in ['r', 'g', 'b']
	for i in range(0, 34):
		color = set_color(result[i,3])
		ax.scatter3D(result[i,0], result[i,1], result[i,2], color, depthshade=False)

	plt.title("PCA results")
	ax.set_xlabel("1st principal component")
	ax.set_ylabel("2nd principal component")
	ax.set_zlabel("3rd principal component")

	# show plot
	plt.show()

def main():
	node, dim = get_data()
	dim = do_pca(dim)

	kmeans = KMeans(n_clusters=3)
	kmeans.fit(dim)
	# print(type(kmeans.labels_))

	result = np.copy(dim)
	# print(np.shape(result), np.shape(kmeans.labels_))
	values = np.array([kmeans.labels_]).transpose()
	result = np.append(result, values, axis=1)
	# result.append() = kmeans.labels_
	# print(np.shape(dim), np.shape(kmeans.labels_))
	# print(np.shape(result))
	# print(np.array([kmeans.labels_]).transpose())

	print_kmeans_graph(result)
	# print_pca_graph(dim)

if __name__ == "__main__":
	main()