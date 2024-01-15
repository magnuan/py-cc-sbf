# pywbms
Python3 CloudCompare SBF file parser

### Usage

```python
from py_cc_sbf import CcSbf
#Read in data file
f_in = CcSbf('some_file.sbf')
meta,data,offset = f_in.read_raw()

#Calculate and append a derived value (swath_offset)
range = data[:,meta.index('range')] 
teta  = data[:,meta.index('teta')]
swath_offset =  range * np.sin(np.rad2deg(teta)])
data = np.concatenate([data.T,[swath_offset]]).T
meta.append('swath_offset')

#Write modified data to new file
f_out = CcSbf('some_new_file.sbf')
f_out.write_raw(meta,data,offset)

```


### Packaging instructions
https://packaging.python.org/en/latest/tutorials/packaging-projects/

