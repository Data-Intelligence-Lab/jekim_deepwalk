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

def make_graph(df):
	index = df['C'].value_counts().index.to_numpy()
	# print(df[df['C'] == 'Soldier'])
	call_adj = count_call(df, index)
	context_adj = count_context(df, index)
	# print(index, len(index), "\n\n***\n")
	print("call\n",call_adj, "\n\n***\n")
	print("context\n", context_adj)
	# np.savetxt("context.txt", np.int64(context_adj))
	"""
	node2vec 보기
	방향성, weight
	"""

def main():
	df = cleaning()
	make_graph(df)

if __name__ == "__main__":
	main()