from grep import grep
import pandas
import glob
import time
import re
import pickle

"""
@author sophie
This executable does the following:
	1. Greps for "E.O." in all ddrs docs, saving to file
	2. Cleans and reads in this data to DataFrame
	3. Also creates dataframe of E.O. metadata

Files needed:
	1. paths.pickle (pickled filepaths of all docs. If not, run 
		'paths = glob.glob("DDRS/ddrs/(0-9)*/*"))'
	2. eo_meta.csv, EO metadata file

Output:
	1. eo_grep.csv, simple grep of "E.O." results
	2. eo_table.csv, ddrs EO data. Columns:
		doc_id, page, eo_id, sec

"""

#paths = glob.glob("DDRS/ddrs/(0-9)*/*")
paths = pickle.load(open('paths.pickle', 'r'))
grep(paths, "E.O.", open('eo_table.csv', 'w'))

## regex patterns ###
path_patt = re.compile('DDRS/ddrs/(\d+)/(\d+)\.txt')
text_patt = re.compile('E\.O\.\s*(.*)[\s\.,]*(.*)')

eo_patt = re.compile('.*E\.O\.\s*(\d*).*')
sec_patt = re.compile('.*(sec|sac)\.*\s*(.*)', re.IGNORECASE)

# write to file
fout = open("eo_table.csv", 'w')
fout.write("path\tdoc_id\tpage\teo_id\tsec\n")

for line in open("eo_grep.csv", "r"):
    m = path_patt.match(line)
    path = m.group()
    doc_id = m.group(1)
    page = m.group(2)

    m = eo_patt.match(line)
    eo_id = ""
    if m:
        eo_id = m.group(1)

    m = sec_patt.match(line)
    sec = ""
    if m:
        sec = m.group(2)

    fout.write(path + "\t" + doc_id + "\t" + page + "\t" + eo_id + "\t" + sec + "\n")
    
fout.close()


## if you want to load the dataframe:
doc_eo = pandas.DataFrame.from_csv('eo_table.csv', sep="\t")
# example: filter by exec order id = 12356
print doc_eo[doc_eo.eo_id == 12356].head

## load the eo_meta.csv with meta data
eo_meta = pandas.DataFrame.from_csv('eo_meta.csv')
# sample
print eo_meta.head()
