#!/usr/bin/env python3
# Copyright 2024
# Magnus Andersen
# Norbit Subsea AS

from struct import pack, unpack
import numpy as np


class CcSbf:
    def __init__(self, filename=None, **kwargs):
        self.meta_filename = filename
        self.data_filename = filename+'.data'
        try:
            self.read_raw_meta()
        except FileNotFoundError:
            #File not found, open for writing
            pass

    def read_raw_meta(self):
        #Read meta data from .sbf file
        with  open(self.meta_filename,'r') as f:
            #Line 1 [SBF] tag
            l = f.readline()
            if l.strip()!='[SBF]':
                raise ValueError("Input file must start with [SBF]")
            d = {}
            for l in f:
                tag = l.split('=')[0]
                val = l.split('=')[1]
                d[tag] = val
        
            self.global_shift = [float(val) for val in d['GlobalShift'].split(',')]
            if  len(self.global_shift) != 3:
                raise ValueError("GlobalShift must specify 3 values")
            self.points = int( d['Points'])
            self.sf_count = int( d['SFCount'])

            sf = ['X','Y','Z']      #X,Y,Z is implicit
            for ix in range(self.sf_count):
                sf.append(d['SF%d'%(ix+1)].strip())
            self.fields = sf

    def read_raw(self,cnt=-1):
        """ Read raw data from file. All data including XYZ in one 2D array of 32-bit floats.
        XYZ Offset value as separate array
        Input:
            cnt: Max number of smples to read, -1 for all
        Returns:
            fields: list (len=M) of field names
            data :  2D array (N x M) of 32-bit values. N datapoints consisting of position (X,Y,Z) and M-3 scalar field values 
            xyz_offset: 1D array (len=3) Offset values to be added to the X,Y and Z values in data """
        #Read raw data from .sbf.data file
        with open(self.data_filename,'rb') as f:
            #Read out binary header and sanity check
            header = f.read(64)
            m1,m2,no_pt,no_sf,offset_x,offset_y, offset_z = unpack('>2BQH3d28x',header)
            if not((m1==42) and (m2==42)):
                raise ValueError("Data file does not start with the magic numbers 42,42")
            if no_pt != self.points:
                raise ValueError("Number of points in binary file does not match that of meta file")
            if no_sf != len(self.fields)-3:
                raise ValueError("Number of scalar fields in binary file does not match that of meta file")
            if False:
                if abs(offset_x + self.global_shift[0])>1e-3:
                    print((offset_x,offset_y, offset_z, self.global_shift))
                    raise Warning("Mismatch between X offset in meta and data file")
                if abs(offset_y + self.global_shift[1])>1e-3:
                    raise Warning("Mismatch between Y offset in meta and data file")
                if abs(offset_z + self.global_shift[2])>1e-3:
                    raise Warning("Mismatch between Z offset in meta and data file")
            if cnt<0:
                d = np.fromfile(f, dtype='>f4')
                if len(d) != no_pt*(no_sf+3):
                    raise ValueError("Data file does not contain the correct amount of data")
                raw_data = d.reshape(no_pt,no_sf+3)
            elif cnt==0:
                raw_data = np.array([])
            else:
                d = np.fromfile(f, dtype='>f4',count=cnt*(no_sf+3))
                if len(d) !=cnt*(no_sf+3):
                    raise ValueError("Data file does not contain the correct amount of data")
                raw_data = d.reshape(cnt,no_sf+3)
            xyz_offset = [offset_x,offset_y,offset_z]
        return self.fields, raw_data, xyz_offset
    
    def read_raw_sample(self,index):
        #Read raw data from .sbf.data file
        with open(self.data_filename,'rb') as f:
            header_len = 64
            sample_len = 4*(self.sf_count+3)
            d = np.fromfile(f, dtype='>f4',count=(self.sf_count+3),offset=header_len+(index*sample_len))
        return d
    
    # Read raw data from field. All data including XYZ in one 2D array of 32-bit floats.
    # XYZ Offset value as separate array
    def read(self):
        """ Read data from file. 
        Append offset and return XYZ values as 64-bit floats, and scalar fields as 32-bit floats
        Returns:
            fields: list of field names, length is equal to number of scalar fields nSF
            pos: 2D array (N x 3) of 64-bit values. Kartesian coordinates of each data point.
            sf : 2D array (N x nSF) of 32-bit values. Scalar fields for each data point """
        fields, raw_data, xyz_offset = self.read_raw()
        pos = raw_data[:,:3].astype('float64')+xyz_offset
        sf = raw_data[:,3:]
        return fields[3:], pos, sf


    def write_raw(self,fields,raw_data, xyz_offset, force_no_pt=None):
        """ Write raw data to file. All data including XYZ in one 2D array of 32-bit floats.
        XYZ Offset value as separate array
        Input:
            fields: list (len=M) of field names
            data :  2D array (N x M) of 32-bit values. N datapoints consisting of position (X,Y,Z) and M-3 scalar field values 
            xyz_offset: 1D array (len=3) Offset values to be added to the X,Y and Z values in data """
        if force_no_pt!=None:
            self.points = force_no_pt
        else:
            self.points = raw_data.shape[0]
        self.sf_count = raw_data.shape[1]-3
        #Write meta data to .sbf file
        with  open(self.meta_filename,'w') as f:
            #Line 1 [SBF] tag
            f.write('[SBF]\n')
            f.write('Points=%d\n'%(self.points))
            f.write('GlobalShift=%f,%f,%f\n'%(-xyz_offset[0],-xyz_offset[1],-xyz_offset[2]))
            f.write('SFCount=%d\n'%(self.sf_count))
            for ix,field in enumerate(fields[3:]):
                f.write('SF%d=%s\n'%(ix+1,field))
            
        with open(self.data_filename,'wb') as f:
            f.write(pack('>2BQH3d28x', 42, 42, self.points, self.sf_count, xyz_offset[0], xyz_offset[1], xyz_offset[2]))
            f.write(np.ascontiguousarray(np.ndarray.flatten(raw_data), dtype='>f4'))
    
    def append_raw(self,raw_data):
        """ Append raw data to file. All data including XYZ in one 2D array of 32-bit floats.
        Input:
            data :  2D array (N x M) of 32-bit values. N datapoints consisting of position (X,Y,Z) and M-3 scalar field values 
            """
            
        with open(self.data_filename,'ab') as f:
            f.write(np.ascontiguousarray(np.ndarray.flatten(raw_data), dtype='>f4'))
   
    def write(self,fields,pos,sf):
        """ Write data to file. 
        Set offset to mid dataset, and scalar fields as 32-bit floats
        Input:
            fields: list of field names, length is equal to number of scalar fields nSF
            pos: 2D array (N x 3) of 64-bit values. Kartesian coordinates of each data point.
            sf : 2D array (N x nSF) of 32-bit values. Scalar fields for each data point """
        xoff = (pos[:,0].max() + pos[:,0].min())/2
        yoff = (pos[:,1].max() + pos[:,1].min())/2
        zoff = (pos[:,2].max() + pos[:,2].min())/2
        xyz_offset = [xoff, yoff, zoff]
        pos32 = (pos-xyz_offset).astype('float32')
        sf32 = sf.astype('float32')
        data = np.concatenate([pos32,sf32],axis=1)
        self.write_raw(['X','Y','Z']+fields, data, xyz_offset)
        
