import sys
import argparse
from py_cc_sbf import CcSbf
import numpy as np

parser = argparse.ArgumentParser(description='Concatenate multiple SBF files into a single one. All input files needs to have same fields',
        epilog='''Example:
        ''',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument ('-i','--input', action='store',dest='input', nargs='*', default=[],help='Input SBF files')
parser.add_argument ('-o','--output', action='store',dest='output', nargs=1, default=[],help='Output file')

args = parser.parse_args()


fnames = args.input
    
datafile_sets = []
total_points = 0
for fname in fnames:
    datafile = CcSbf(fname)
    datafile_sets.append(datafile)
    total_points += datafile.points
    print("Reading input data from %s, %d points" % (fname,datafile.points))
print("Total ponts in files = %d, fields = %d "%(total_points,datafile.sf_count))

f_out = CcSbf(args.output[0])

for ix,datafile in enumerate(datafile_sets):
    fields,d,offset = datafile.read_raw()
    if ix==0:
        fields0 = fields
        offset0 = offset
        f_out.write_raw(fields,d,offset,force_no_pt=total_points)
    else:
        if (fields != fields0):
            print('All input files must have the ecxact same fields')
            sys.exit(-1)
    
        offset_diff = np.array(offset) - np.array(offset0)
        d[:,:3] = d[:,:3] + offset_diff
        f_out.append_raw(d)

