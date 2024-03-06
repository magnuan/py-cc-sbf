from py_cc_sbf import CcSbf
import numpy as np
import sys

f_in = CcSbf('/home/magnuan/data/20240105_cloud_classification/2023-02-28-11_42_06_classified.sbf')
fields,data,offset = f_in.read_raw()

#Discard some data based on some criteria
if True:
    gix = np.where(data[:,fields.index('val')]>0.1)[0]
    data = data[gix]
    #gix = np.where( (data[:,fields.index('pingnumber')]>32100)*(data[:,fields.index('pingnumber')]<32500)  )[0]
    #data = data[gix]

#### Modify the data a bit to ease processing
pingnumber0 = data[:,fields.index('pingnumber')].min()
if True:
    #Start pingnumbere indexing at 0
    data[:,fields.index('pingnumber')] -= pingnumber0
    #Positive downwards
    data[:,fields.index('Z')] *= -1
    #Adjust minimum depth to 1m
    data[:,fields.index('Z')] -= data[:,fields.index('Z')].min()-1
    #Adjust intensity by a fixed scaling
    #data[:,fields.index('val')] *= 0.1
    data[:,fields.index('val')] = np.sqrt(data[:,fields.index('val')])

##### Sort on ping and beam, to prevent som odd sorting of data
if True:  
    beam = data[:,fields.index('beam')].astype('int32')
    pingnumber = data[:,fields.index('pingnumber')].astype('int32')
    nBeam = beam.max()+1
    sounding_index = pingnumber*nBeam + beam
    six = np.argsort(sounding_index)
    data = data[six]






beam = data[:,fields.index('beam')].astype('int32')
pingnumber = data[:,fields.index('pingnumber')].astype('int32')
data_range = data[:,fields.index('range')]
labels = data[:,fields.index('class')].astype('uint8')

nBeam = beam.max()+1
nPing = pingnumber.max()+1
nSamples = len(data)

#These are the fields we extract directly as a property of each sounding
data_1d_fields = ['val','Z','range','teta']
#These are the fields for which we process the neighborhood of each sounding
data_2d_fields = ['Z']

#Put selected data into a regular 2D grid, based on ping and beam number, to extraxct neighborhood from
index_2d = pingnumber*nBeam + beam
data_2d = {}

for f in data_2d_fields:
    data_2d[f] =  np.zeros(nBeam*nPing)
    data_2d[f][index_2d] = data[:,fields.index(f)]
    data_2d[f] = np.reshape(data_2d[f],(nPing,nBeam))
#Also generate a validity map for 2d data
data_2d_valid = np.zeros(nBeam*nPing)
data_2d_valid[index_2d]=1
data_2d_valid = np.reshape(data_2d_valid,(nPing,nBeam))

#Generate augmented data tensor for each sample
fix_1d = [fields.index(k) for k in data_1d_fields]

#Sample index
six = 1000

#Neghborhood size, at each zoom level we generate a 8x8 image from each 2D data set
neighborhood_size = 10
nSz = neighborhood_size//2
#Zoomlevels
zoom_levels = [1,2,4,8,16]

#Padding, pad 2d data to easily extract neghborhood without worying about edges
padding = max(zoom_levels) * nSz

data_2d_valid_padded = np.pad(data_2d_valid,padding)
data_2d_padded = {}
for f in data_2d_fields:
    data_2d_padded[f] = np.pad(data_2d[f],padding)

n_neighborhood = len(zoom_levels)*len(data_2d_fields)
n_sample_params = n_neighborhood * neighborhood_size**2 + len(data_1d_fields)

#sample_indexes = np.array(range(100000,500000,1))
sample_indexes = np.array(range(len(data)))
sample_indexes = np.where(beam==128)[0]
pingnumber_range = [32200, 32500]
sample_indexes = np.where( (pingnumber>=(pingnumber_range[0]-pingnumber0)) * (pingnumber<=(pingnumber_range[1]-pingnumber0))   )[0]

D = np.zeros((len(sample_indexes),n_sample_params),dtype='float32')

d_2d = np.zeros((len(zoom_levels)*len(data_2d_fields),neighborhood_size,neighborhood_size))

print("Augmenting dataset")
for sixx,six in enumerate(sample_indexes):
    if (sixx%100==0):
        sys.stdout.write('\r%d/%d'%(sixx,len(sample_indexes)))
        sys.stdout.flush()
    #Extracting 1d field data for each sample
    d_1d = data[six,fix_1d]
   
    #Collecting neighborhood data for each sample
    pix = pingnumber[six]+padding   #Ping number for given sample
    bix = beam[six]+padding         #Beam number for given sample
    set_ix=0
    for f_2d in data_2d_fields:     #Iterate over all neighborhood fields
        if f_2d=='Z':
            sample_val = data_2d_padded[f_2d][pix,bix] #Field data in sample itself
            for zl in zoom_levels:          #Iterate over zoom levels
                neighborhood = data_2d_padded[f_2d][ pix-(zl*nSz)+(zl//2):pix+(zl*nSz):zl,  bix-(zl*nSz)+(zl//2):bix+(zl*nSz):zl] #Field data in neighborhood
                neighborhood = (neighborhood-sample_val)*(neighborhood!=0)*(10)/data_range[six]  #Depth data normalization
                d_2d[set_ix] = neighborhood
                set_ix += 1
        elif f_2d=='val':
            sample_val = data_2d_padded[f_2d][pix,bix] #Field data in sample itself
            for zl in zoom_levels:          #Iterate over zoom levels
                neighborhood = data_2d_padded[f_2d][ pix-(zl*nSz)+(zl//2):pix+(zl*nSz):zl,  bix-(zl*nSz)+(zl//2):bix+(zl*nSz):zl] #Field data in neighborhood
                neighborhood = neighborhood*data_range[six]/1000                                    #Intensity data normalization
                d_2d[set_ix] = neighborhood
                set_ix += 1
        else:
            for zl in zoom_levels:          #Iterate over zoom levels
                neighborhood = data_2d_padded[f_2d][ pix-(zl*nSz)+(zl//2):pix+(zl*nSz):zl,  bix-(zl*nSz)+(zl//2):bix+(zl*nSz):zl] #Field data in neighborhood
                d_2d[set_ix] = neighborhood
                set_ix += 1
    #d_2d /= data_range[six]
    #Concatenate 1D and 2D tensor data
    d =  np.concatenate((d_1d,np.ndarray.flatten(d_2d)))
    D[sixx,:] = d
print('')

out_filename = 'survey_dataset1'
print("Writing outout to %s.sbf and %s_augmented.pickle"%(out_filename,out_filename))
#Write selected data back to SBF file
f_out = CcSbf(out_filename+'.sbf')
f_out.write_raw(fields,data[sample_indexes,:],offset)

#Write augmented data to pickle file
dataset = {'data':D, 'labels':labels[sample_indexes]}
from pickle import dump
with open(out_filename+'_augmented.pickle','wb') as f:
    dump(dataset,f)



if False:
    #Plot single neighborhood set
    
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    first = True
    for q in range(0,len(D),1):
        nbh =  np.reshape(D[q][len(data_1d_fields):],(n_neighborhood,neighborhood_size,neighborhood_size))
        if first:
            pl = plt.imshow(np.reshape(np.rollaxis(nbh,2),(neighborhood_size,neighborhood_size*n_neighborhood)),aspect='auto',vmin=-3, vmax=3)
            first=False
        else:
            pl.set_data(np.reshape(np.rollaxis(nbh,2),(neighborhood_size,neighborhood_size*n_neighborhood)))
        axs.set_title("Ping=%d, Beam=%d, Class=%d"%(pingnumber[sample_indexes][q], beam[sample_indexes][q], labels[sample_indexes][q]))
        plt.show()
        input()
