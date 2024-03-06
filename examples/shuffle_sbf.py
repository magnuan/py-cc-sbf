import sys
import argparse
from py_cc_sbf import CcSbf
import numpy as np

parser = argparse.ArgumentParser(description='Shuffle samples in SBF file randomly',
        epilog='''Example:
        ''',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument ('-i','--input', action='store',dest='input', nargs=1, default=[],help='Input SBF file')
parser.add_argument ('-o','--output', action='store',dest='output', nargs=1, default=[],help='Output SBF file')

args = parser.parse_args()

f_in = CcSbf(args.input[0])
print("Total ponts in file = %d, fields = %d "%(f_in.points,f_in.sf_count))

f_out = CcSbf(args.output[0])

fields,d,offset = f_in.read_raw(cnt=0)

shuffle_order = np.array(range(f_in.points))
np.random.shuffle(shuffle_order)

print_dec = f_in.points//100

for ix,n in enumerate(shuffle_order):
    if(ix%print_dec==0):
        print("%.0f%%"%(ix*100//f_in.points))

    d = f_in.read_raw_sample(n)
    if ix==0:
        f_out.write_raw(f_in.fields,np.array([d]),offset,force_no_pt=f_in.points)  #Write header and first data point
    else:
        f_out.append_raw(d)
