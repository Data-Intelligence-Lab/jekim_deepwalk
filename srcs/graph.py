from locale import normalize
from turtle import end_fill
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

def get_data():
	filename = "../data/ff7-script.csv"
	df = pd.read_csv(filename)

	c = df.iloc[:,[1,2]]
	c = c[~c['Character'].str.contains('On|screen|Grey|text|NPC')]

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

def count_call(df):
	index = df['C'].value_counts().index.to_numpy()
	# i=int(np.where(index == 'Sephiroth')[0])
	call_adj = np.zeros((len(index), len(index)))
	dial = df['D']
	for i in dial:
		mention = is_mentioned(index, i)
		# if mention != 0:
			# row = df[dial == i].index
			# print(row)
			# col = int(np.where(index == mention)[0])
			# call_adj[row, col] += 1
	print(call_adj)

# def count_context(df):

# def make_graph(data):
# 	window = 5

def main():
	df = cleaning()
	count_call(df)

if __name__ == "__main__":
	main()