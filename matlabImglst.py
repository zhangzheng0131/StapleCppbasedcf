from os import listdir
from os.path import isfile, join
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-d', action='store',dest='mypath',help='image directory')
parser.add_argument('-f', action='store',dest='filepath',help='where to output' ,default='./imglst.lst')
parser.add_argument('-s', action='store',dest='start',help='start frame')
parser.add_argument('-e', action='store',dest='end',help='end frame')
parser.add_argument('-n', action='store',dest='nz',help='number of zero to pad')
parser.add_argument('-x', action='store',dest='ext',help='file ext')
parser.add_argument('-r', action='store',dest='rect',help='init rect')
results = parser.parse_args()

mypath = results.mypath
gg=results.rect

sf = int(results.start)
ef = int(results.end)
nz = int(results.nz)
ext = results.ext

# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f)) and f[0]!='.']
# onlyfiles = sorted(onlyfiles)
# import pdb
# pdb.set_trace()
# onlyfiles = onlyfiles[sf:ef]

f = open(results.filepath,'w')

for i in range(ef-sf+1):
    ff = str(i+sf).zfill(nz) +'.'+ ext
    f.write(join(mypath,ff))
    f.write(',')
    gg = gg.replace(' ',',')
    gg = gg.replace('\t',',')
    f.write(gg)
    f.write('\n')
f.close()

