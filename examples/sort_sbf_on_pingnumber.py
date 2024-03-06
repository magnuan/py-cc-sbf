from py_cc_sbf import CcSbf
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='Sort a SBF file on "pingnumber" field',
        epilog='''Example:
        ''',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument ('-i','--input', action='store',dest='inn', nargs=1, default=[],help='Input file')
parser.add_argument ('-o','--output', action='store',dest='out', nargs=1, default=[],help='Output file')

args = parser.parse_args()

f_in = CcSbf(args.inn[0])

print("Reading data from %s" % (args.inn[0]))
print("Writing sorted data to %s" % (args.out[0]))

sort_field_name = 'pingnumber'
fields,data,offset = f_in.read_raw()

if not( (sort_field_name in fields) ):
    print("Input SBF does not contain %s field"%(sort_field_name))
    sys.exit(-1)

sort_field = data[:,fields.index(sort_field_name)]
six = np.argsort(sort_field)
data = data[six,:]

#Write modified data to new file
f_out = CcSbf(args.out[0])
f_out.write_raw(fields,data,offset)

