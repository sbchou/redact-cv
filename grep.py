import pandas
import glob
import time
import re

## emulates output of grep
def grep(paths, string):
	for f in paths:
		for line in open(f):
			if string in line:
				print(f + ":" + line)

#paths = glob.glob("DDRS/ddrs/[0-9]*/*")
paths = pickle.load(open('paths.pickle', 'r'))
grep(paths, "E.O.") #cat this to a file