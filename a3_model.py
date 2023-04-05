from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim

class Perceptron(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_size, nonlinearity):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, hidden_size)
		if nonlinearity == "relu":
			self.nonlinearity = nn.ReLU()
		elif nonlinearity == "tanh":
			self.nonlinearity = nn.Tanh()
		else:
			self.nonlinearity = nn.Identity()
		self.linear2 = nn.Linear(hidden_size, output_dim)
		self.logsoftmax = nn.LogSoftmax(dim=1)
        
	def forward(self, x):
		x = self.linear1(x)
		x = self.nonlinearity(x)
		x = self.linear2(x)
		x = self.logsoftmax(x)
		return x

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train and test a model on features.")
	parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
	parser.add_argument("--hidden_size", type=int, default=0, help="Size of the hidden layer.")
	parser.add_argument("--nonlinearity", type=str, choices=["relu", "tanh", "none"], default="none", help="Non-linearity function to use.")
	args = parser.parse_args()
	
	print("Reading {}...".format(args.featurefile))
	df = pd.read_csv(args.featurefile)
	train_df = df.loc[df.iloc[:, -1].str.contains('train')]
	X_train = train_df.iloc[1:, :-2].values
	y_train = train_df.iloc[1:, -2].values
	test_df = df.loc[df.iloc[:, -1].str.contains('test')]
	X_test = test_df.iloc[1:, :-2].values
	y_test = test_df.iloc[1:, -2].values
    
	input_dim = X_train.shape[1]
	output_dim = len(np.unique(y_train))
  
	model = Perceptron(input_dim, output_dim, args.hidden_size, args.nonlinearity)
	le = LabelEncoder()
	y_train_numbers = le.fit_transform(y_train)
	X_train_torch = torch.tensor(X_train, dtype=torch.float32)
	y_train_torch = torch.tensor(y_train_numbers, dtype=torch.long)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	loss_function = nn.NLLLoss()
	model.train()
	for epoch in range(100):
		optimizer.zero_grad()
		logits = model(X_train_torch)
		loss = loss_function(logits, y_train_torch)
		loss.backward()
		optimizer.step()
	model.eval()
	X_test_torch = torch.tensor(X_test, dtype=torch.float32)
	logits = model(X_test_torch)
	probs = logits.argmax(dim=1)
	pred = []
	for number in probs:
		pred.append(np.unique(y_train)[number])
	cm = confusion_matrix(y_test, pred)
	classes = np.unique(y_test)
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
