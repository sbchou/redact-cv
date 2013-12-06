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
"""
def main():
    #paths = glob.glob("DDRS/ddrs/(0-9)*/*")
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('pickled_paths')
    parser.add_argument('grep_out_path')
    parser.add_argument('eo_out_path')
    parser.add_argument('meta_table_path')    

    args = vars(parser.parse_args())

    paths = pickle.load(open('../data/paths.pickle', 'r'))
    #paths = glob.glob("../data/ddrs/*/*")

    with open(args['grep_out_path'], 'w') as grep_out:
        for line in grep(paths, "E.O."):
            grep_out.write(line)
    
    ## regex patterns ###
    path_patt = re.compile('../data/ddrs/(\d+)/(\d+)\.txt')
    text_patt = re.compile('E\.O\.\s*(.*)[\s\.,]*(.*)')

    eo_patt = re.compile('.*E\.O\.\s*(\d*).*')
    sec_patt = re.compile('.*(sec|sac)\.*\s*(.*)', re.IGNORECASE)

    # write cleaned tables to file
    with  open(args['eo_out_path'], 'w') as fout:
        fout.write("number\tpath\tdoc_id\tpage\teo_id\tsec\n")
        i = 0
        for line in open(args['grep_out_path'], 'r'):
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
            fout.write(str(i) + "\t" + path + "\t" + doc_id + "\t" + page + "\t" + eo_id + "\t" + sec + "\n")
            i += 1
   
if __name__ == "__main__":
    main()
