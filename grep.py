"""
@author sophie
function to emulate Unix grep
"""

def grep(paths, string, fout=False):
    """ Performs a search on files with given paths for a
    string. Emulates Unix grep, for strings only.

    Parameters
    ----------
    paths: list of file paths, which are strings
    string: the string you're searching for
    fout: (optional) the open file to write results out to.
    If fout not given, print results to stoudt instead.
    
    Returns
    -------
    None. Either prints results or writes to file.

    """
    for f in paths:
        for line in open(f):
            if string in line:
                if fout:
                    fout.write(f + ":" + line + "\n")
                else:
                    print(f + ":" + line)
    if fout:
        fout.close()
            