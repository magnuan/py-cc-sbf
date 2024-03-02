from py_cc_sbf import CcSbf
import numpy as np
import sys



f_in = CcSbf('/home/magnuan/data/20240105_cloud_classification/2023-02-28-11_42_06.sbf')
f_class = CcSbf('/home/magnuan/data/20240105_cloud_classification/2023-02-28-11_42_06_classified.sbf')

nBeams = 4096   #Assume maximum 4096 beams

fields,data,offset = f_class.read_raw()
if not( ('beam' in fields) and ('pingnumber' in fields) and ('class' in fields) ):
    print("Class input SBF must contain beam, pingnumber and class field")
    sys.exit(-1)
c_id = data[:,fields.index('beam')].astype('u8') + data[:,fields.index('pingnumber')].astype('u8')*nBeams
c_class = data[:,fields.index('class')]


fields,data,offset = f_in.read_raw()
if not( ('beam' in fields) and ('pingnumber' in fields) ):
    print("Input SBF must contain beam and pingnumber field")
    sys.exit(-1)
i_id = data[:,fields.index('beam')].astype('u8') + data[:,fields.index('pingnumber')].astype('u8')*nBeams


append_class_field = not( 'class' in fields)
if append_class_field:
    i_class = 0*data[:,fields.index('beam')]-1  #Set unknown classes to -1
    fields.append('class')
else:
    i_class = data[:,fields.index('class')]     #Keep old classes for datapoints not in class set

#Find mapping of class_id in input_id
six = np.searchsorted(i_id,c_id)
#Set classes of points that exist in class set 
i_class[six] = c_class

if append_class_field:
    data = np.concatenate([data.T,[i_class]]).T
else:
    data[:,fields.index('class')] = i_class


#Write modified data to new file
f_out = CcSbf('/home/magnuan/data/20240105_cloud_classification/2023-02-28-11_42_06_reclassified.sbf')
f_out.write_raw(fields,data,offset)

