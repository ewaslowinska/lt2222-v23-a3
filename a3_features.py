from sklearn.model_selection import train_test_split
import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Convert directories into table.")
	parser.add_argument("inputdir", type=str, help="The root of the author directories.")
	parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
	parser.add_argument("dims", type=int, help="The output feature dimensions.")
	parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")
	
	args = parser.parse_args()
	
	print("Reading {}...".format(args.inputdir))
	
	def author_and_email(inputdir):
		authors_and_bodies = []
		for folder in os.listdir(inputdir)[1:]:
			for fil in os.listdir(inputdir + folder):
				path = inputdir + folder + "/" + fil	
				with open(path) as f:
					email = f.read()
					author = re.findall(r'[a-z]+-[a-z]', path)
					index = email.find('\n\n')
					email_body = email[index+2:]
					index1 = email_body.find('-----')
					email_body1 = re.sub(r'\n', '', email_body[:index1])
					email_body2 = re.sub(r'\t', '', email_body1)
					authors_and_bodies.append((author[0], email_body2))
		return authors_and_bodies

	authors_and_bodies = author_and_email(args.inputdir)

	print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
	vectorizer = CountVectorizer(max_features=args.dims)
	X = vectorizer.fit_transform([body for author, body in authors_and_bodies])
	test_size = args.testsize / 100.0
	X_train, X_test, y_train, y_test = train_test_split(X, [author for author, body in authors_and_bodies], test_size=test_size)
	
	print("Writing to {}...".format(args.outputfile))
	train_df = pd.DataFrame(X_train.toarray())
	train_df['author'] = y_train
	train_df['set'] = 'train'
	test_df = pd.DataFrame(X_test.toarray())
	test_df['author'] = y_test
	test_df['set'] = 'test'
	df = pd.concat([train_df, test_df], ignore_index=True)
	df.to_csv(args.outputfile, index=False, columns=list(range(args.dims)) + ['author'] + ['set'])
	
	print("Done!")
