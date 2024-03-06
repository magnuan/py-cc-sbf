from py_cc_sbf import CcSbf
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='Reads in two SBF files, Copies Classification data from one to another based on beam and pingnumber',
        epilog='''Example:
        ''',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument ('-c','--input_class', action='store',dest='classified', nargs=1, default=[],help='Input file, file to copy classification data from')
parser.add_argument ('-i','--input', action='store',dest='inn', nargs=1, default=[],help='Input file, file to be classified')
parser.add_argument ('-o','--output', action='store',dest='out', nargs=1, default=[],help='Output file')

args = parser.parse_args()

f_in = CcSbf(args.inn[0])
f_class = CcSbf(args.classified[0])

print("Reading classes from %s" % (args.classified[0]))
print("Reading data from %s" % (args.inn[0]))
print("Writing classified data to %s" % (args.out[0]))

nBeams = 4096   #Assume maximum 4096 beams

fields,data,offset = f_class.read_raw()
if not( ('Classification' in fields) ):
    print("Class input SBF must contain  Classification field")
    sys.exit(-1)


if ( ('beam' in fields) and ('pingnumber' in fields) ):
    print("Using  beam and  pingnumber for sounding ID")
    c_id = data[:,fields.index('beam')].astype('u8') + data[:,fields.index('pingnumber')].astype('u8')*nBeams
    id_source = 'beam pingnumber'
elif ( ('teta' in fields) and ('range' in fields) ):  #This does not seem to work
    print("Using  teta and  range for sounding ID")
    c_id = data[:,fields.index('teta')] + data[:,fields.index('range')]*180
    id_source = 'teta range'
else:
    print("No valid data for sounding ID in class input file")
    sys.exit(-1)

c_class = data[:,fields.index('Classification')]


fields,data,offset = f_in.read_raw()
if id_source == 'beam pingnumber':
    if not( ('beam' in fields) and ('pingnumber' in fields) ):
        print("Input SBF must contain beam and pingnumber field")
        sys.exit(-1)
    i_id = data[:,fields.index('beam')].astype('u8') + data[:,fields.index('pingnumber')].astype('u8')*nBeams
if id_source == 'teta range':
    if not( ('teta' in fields) and ('range' in fields) ):
        print("Input SBF must contain teta and range field")
        sys.exit(-1)
    i_id = data[:,fields.index('teta')] + data[:,fields.index('range')]*180




append_class_field = not( 'Classification' in fields)
if append_class_field:
    i_class = 0*data[:,fields.index('beam')]-1  #Set unknown classes to -1
    fields.append('Classification')
else:
    i_class = data[:,fields.index('Classification')]     #Keep old Classificationes for datapoints not in class set

print("Number of datapoints in class input = %d" %len(c_id))
print("Number of datapoints in data input = %d" %len(i_id))
#Find mapping of class_id in input_id
six = np.searchsorted(i_id,c_id)
#Set classes of points that exist in class set 
i_class[six] = c_class

if append_class_field:
    data = np.concatenate([data.T,[i_class]]).T
else:
    data[:,fields.index('Classification')] = i_class
c_id = data[:,fields.index('beam')].astype('u8') + data[:,fields.index('pingnumber')].astype('u8')*nBeams


#Write modified data to new file
f_out = CcSbf(args.out[0])
f_out.write_raw(fields,data,offset)

