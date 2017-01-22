from os import listdir
from os.path import isfile, join
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-d', action='store',dest='mypath',help='image directory')
parser.add_argument('-f', action='store',dest='filepath',help='where to output' ,default='./imglst.lst')

parser.add_argument('-r', action='store',dest='rect',help='init rect')
results = parser.parse_args()

mypath = results.mypath
gg=results.rect

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f)) and f[0]!='.']
onlyfiles = sorted(onlyfiles)

f = open(results.filepath,'w')

for ff in onlyfiles:
    f.write(join(mypath,ff))
    f.write(',')
    gg = gg.replace(' ',',')
    gg = gg.replace('\t',',')
    f.write(gg)
    f.write('\n')
f.close()

