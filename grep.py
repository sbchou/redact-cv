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
            
#paths = glob.glob("DDRS/ddrs/(0-9)*/*")
paths = pickle.load(open('paths.pickle', 'r'))
grep(paths, "E.O.") #cat this to a file

## regex patterns ###
path_patt = re.compile('DDRS/ddrs/(\d+)/(\d+)\.txt')
text_patt = re.compile('E\.O\.\s*(.*)[\s\.,]*(.*)')

eo_patt = re.compile('.*E\.O\.\s*(\d*).*')
sec_patt = re.compile('.*(sec|sac)\.*\s*(.*)', re.IGNORECASE)

# write to file
fout = open("eo_table.csv", 'w')
fout.write("path\tdoc_id\tpage\teo_id\tsec\n")

for line in open("EO_grep.csv", "r"):
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
df = pandas.DataFrame.from_csv('eo_table.csv', sep="\t")

