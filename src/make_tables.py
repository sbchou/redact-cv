from grep import grep
import pandas
import glob
import time
import re
import pickle
import sys
import argparse

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
def main():
    #paths = glob.glob("DDRS/ddrs/(0-9)*/*")
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('pickled_paths')
    parser.add_argument('grep_out_path')
    parser.add_argument('eo_out_path')
    parser.add_argument('meta_table_path')    

    args = vars(parser.parse_args())

    #paths = pickle.load(open('data/paths.pickle', 'r'))
    paths = glob.glob("../data/ddrs/311*/*")
    
    with open(args['grep_out_path'], 'w') as grep_out:
        for line in grep(paths, "E.O."):
            print >> grep_out, line

    
    ## regex patterns ###
    path_patt = re.compile('../data/ddrs/(\d+)/(\d+)\.txt')
    text_patt = re.compile('E\.O\.\s*(.*)[\s\.,]*(.*)')

    eo_patt = re.compile('.*E\.O\.\s*(\d*).*')
    sec_patt = re.compile('.*(sec|sac)\.*\s*(.*)', re.IGNORECASE)

    # write to file
    fout = open(args['eo_out_path'], 'w')
    fout.write("path\tdoc_id\tpage\teo_id\tsec\n")

    for line in open(args['grep_out_path'], 'r'):
        if line.strip():
            m = path_patt.match(line)
            print m
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

    """

    ## if you want to load the dataframe:
    doc_eo = pandas.DataFrame.from_csv('data/eo_table.csv', sep="\t")
    # example: filter by exec order id = 12356
    print doc_eo[doc_eo.eo_id == 12356].head

    ## load the eo_meta.csv with meta data
    eo_meta = pandas.DataFrame.from_csv('data/eo_meta.csv')
    # sample
    print eo_meta.head()
    """

if __name__ == "__main__":
    main()
