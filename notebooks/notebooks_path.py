import os
import sys

def get_notebook_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_tld():
    return os.path.dirname(get_notebook_dir())


def include_packages():
    tld = get_tld()
    sys.path.append(os.path.join(tld, 'playground')) # this lets us import playground


