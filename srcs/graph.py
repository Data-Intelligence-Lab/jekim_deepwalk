from locale import normalize
from turtle import end_fill
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

def get_data():
	filename = "../data/ff7-script.csv"
	df = pd.read_csv(filename)

	c = df.iloc[:,[1,2]]
	c = c[~c['Character'].str.contains('On|screen|Grey|text|NPC|Guard')]

	data = c.to_numpy()
	return data

def cleaning():
	data = get_data()
	for i in range(0, len(data)):
		data[i,0] = data[i,0].strip().replace(" ","")
	df = pd.DataFrame(data, columns=['C', 'D'])

	value = df['C'].value_counts()
	index = df['C'].value_counts().index[value < 10] # appearance frequency
	
	for i in index:
		df = df.drop(df[df['C'] == i].index)
	df.reset_index(drop=True, inplace=True)

	return df

def is_mentioned(index, str):
	for i in index:
		if i in str.replace(" ",""):
			return i
	return 0

def count_call(df, index):
	call_adj = np.zeros((len(index), len(index)))
	dial = df['D']
	d_i = dial.index.to_numpy()
	for i in d_i:
		mention = is_mentioned(index, dial[i])
		if mention != 0:
			row = int(np.where(index == df.loc[i, 'C'])[0]) # who
			col = int(np.where(index == mention)[0]) # mentioned
			call_adj[row, col] += 1
	for j in range(0, len(index)):
		call_adj[j, j] = 0
	return call_adj

def count_context(df, index):
	context_adj = np.zeros((len(index), len(index)))
	window = 5 # odd number
	half = int(window / 2)
	cha = df['C']
	cha_i = cha.index.to_numpy()
	for i in cha_i:
		row = int(np.where(index == cha[i])[0])
		ran = range(i - half, i + half + 1)
		if i < half:
			ran = range(0, i + half + 1)
		elif i + half + 1 >= len(cha_i):
			ran = range(i - half, len(cha_i))
		for j in ran:
			col = int(np.where(index == cha[j])[0])
			context_adj[row, col] += 1
		context_adj[row, row] = 0
	return context_adj

def make_adjlist(context_adj):
	threshold = np.median(context_adj[np.nonzero(context_adj)]) # nonzero median value = threshold
	# mean = np.mean(context_adj[np.nonzero(context_adj)])
	# print(threshold, mean)
	# np.savetxt("nonzero_call.txt", context_adj[np.nonzero(context_adj)], fmt='%d')
	adjmat = np.where(context_adj < threshold, 0, 1)
	# print(len(adjmat[np.nonzero(adjmat)]), len(context_adj[np.nonzero(context_adj)]))
	# # np.savetxt("adjmat.txt", adjmat, fmt='%d')
	# with open("../graphs/undir_list.adjlist",'w') as f:
	# 	for i in range(0, len(adjmat)):
	# 		a = np.array(np.nonzero(adjmat[i])).flatten().tolist()
	# 		s = " ".join(str(e) for e in a)
	# 		f.write(str(i) + " " + s)
	# 		f.write("\n")

def make_directed(call_adj, flag):
	threshold = np.median(call_adj[np.nonzero(call_adj)])
	if flag == 1:
		""" 1. directed, unweighted,
			threshold applied = nonzero median """
		dir_adjmat = np.where(call_adj < threshold, 0, 1)
	elif flag == 2:
		""" 2. directed, unweighted,
			no threshold """		
		dir_adjmat = np.where(call_adj > 0, 1, 0)
	elif flag == 3:
		""" 3. directed, weighted """	
		dir_adjmat = call_adj
	s = "../graphs/dir_mat" + str(flag) + ".adjmat"
	# np.savetxt(s, dir_adjmat, fmt='%d')
	with open("../graphs/dir_list.adjlist",'w') as f:
		for i in range(0, len(dir_adjmat)):
			a = np.array(np.nonzero(dir_adjmat[i])).flatten().tolist()
			s = " ".join(str(e) for e in a)
			f.write(str(i) + " " + s)
			f.write("\n")

def make_weighted(context_adj, call_adj):
	cont_ths = np.median(context_adj[np.nonzero(context_adj)])
	call_ths = np.median(call_adj[np.nonzero(call_adj)])

	adjmat = np.where(context_adj <= cont_ths, 0, 1.0)
	dir_adjmat = np.where(call_adj <= call_ths, 0, 0.5)

	for i in range(0, len(dir_adjmat)):
		for j in range(i + 1, len(dir_adjmat)):
			w = dir_adjmat[i, j] + dir_adjmat[j, i]
			adjmat[i, j] += w
			adjmat[j, i] += w
	print(adjmat)
	# np.savetxt("adjmat", adjmat, fmt='%f')


def make_graph(df):
	index = df['C'].value_counts().index.to_numpy()
	# np.savetxt("../graphs/index", index, fmt='%s')
	call_adj = count_call(df, index)
	context_adj = count_context(df, index)
	make_weighted(context_adj, call_adj)

	# make_adjlist(context_adj)
	# make_directed(call_adj, flag=1) # use flag=1
	print("finished")

def main():
	df = cleaning()
	make_graph(df)

if __name__ == "__main__":
	main()