import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 

def get_data():
	"""
	get the node and latent representation data 
	from the output of Deepwalk algorithm
	Skipgram format -> node and dimension in first line are removed
	latent representation values are standardized
	"""

	filename = "../embeddings/undir.embeddings"
	df = pd.read_csv(filename, sep=' ', header=None)
	node = df.iloc[:,0].to_numpy()
	dim = df.iloc[:,1:] # pandas dataframe
	std_dim = StandardScaler().fit_transform(dim) # numpy array
	return [node, std_dim]

def do_pca(std_dim):
	"""
	PCA by sklearn
	n_components = 3 : accumulated variance ratio is 0.9845891301350316
	"""

	pca = PCA(n_components=3)
	p_comp = pca.fit_transform(std_dim)

	print(pca.explained_variance_ratio_)
	print(sum(pca.explained_variance_ratio_))

	return p_comp

def print_pca_graph(data):
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	ax.scatter3D(data[:,0], data[:,1], data[:,2], color = "green")

	plt.title("PCA results")
	ax.set_xlabel("1st principal component")
	ax.set_ylabel("2nd principal component")
	ax.set_zlabel("3rd principal component")

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

	for i in range(0, 34):
		c = set_color(result[i,3])
		ax.scatter3D(result[i,0], result[i,1], result[i,2], color=c, depthshade=False)

	plt.title("Kmeans results")
	ax.set_xlabel("1st principal component")
	ax.set_ylabel("2nd principal component")
	ax.set_zlabel("3rd principal component")

	plt.show()

def do_kmeans(dim):
	"""
	K-means clustering by sklearn
	n_clusters = 3 : by heuristic
	return -> numpy array result : k-means result labels are appended into array dim
	"""
	kmeans = KMeans(n_clusters=3)
	kmeans.fit(dim)

	result = np.copy(dim)
	values = np.array([kmeans.labels_]).transpose()
	result = np.append(result, values, axis=1)

	return result

def main():
	node, dim = get_data()
	dim = do_pca(dim)
	# result = do_kmeans(dim)

	print_pca_graph(dim)
	# print_kmeans_graph(result)

if __name__ == "__main__":
	main()