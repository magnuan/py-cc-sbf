#Load training dataset
from pickle import load
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
from py_cc_sbf import CcSbf
import random
import sys
import argparse

parser = argparse.ArgumentParser(description='Train neural network with data from SBF file',
        epilog='''Example:
        ''',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument ('-T','--train', action='store',dest='train', nargs='*', default=[],help='Input SBF file with training / validation data')
parser.add_argument ('-t','--test', action='store',dest='test', nargs=1, default=[],help='Input SBF file with test data')
parser.add_argument ('-i','--input', action='store',dest='model_in', nargs=1, default=[],help='Input model file')
parser.add_argument ('-o','--output', action='store',dest='model_out', nargs=1, default=[],help='Output model file')

args = parser.parse_args()

n_classes = 8
max_test_data = 2000000


has_train_data = len(args.train)>0
has_test_data = len(args.test)>0
has_model = len(args.model_in)>0
write_model = len(args.model_out)>0

if  has_model:  #Read existing model from file
    epochs=1
    model = keras.models.load_model(args.model_in[0])
else:
    print("Training new model from scratch")
    epochs=10
    #Set up new NN model
    model = keras.Sequential([
            layers.Dense(32,activation='relu'),
            layers.Dense(32,activation='relu'),
            layers.Dense(32,activation='relu'),
            layers.Dense(n_classes,activation='softmax')
            ])

    model.compile(  optimizer=keras.optimizers.RMSprop(learning_rate=0.0003),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


def read_super_batch_from_files(fnames,super_batch_size=2000000, valid_split=0):
    f_train_sets = []
    total_points = 0
    for fname in fnames:
        f_train = CcSbf(fname)
        f_train_sets.append(f_train)
        total_points += f_train.points
        print("Reading training data from %s, %d points" % (fname,f_train.points))
    print("Total ponts in files = %d"%total_points)
    data_usage = super_batch_size/total_points
    print("Using %0.1f%% of data"%(data_usage*100))


    data = []
    labels = []
    v_data=[]
    v_labels=[]
    for ix,f_train in enumerate(f_train_sets):
        fields,d,offset = f_train.read_raw()
        #Sort on pingnumber
        pingnumber = d[:,fields.index('pingnumber')]
        six = np.argsort(pingnumber)
        d = d[six,:]
        if (ix==0):             #Check that all files has same fields
            fields0 = fields
        else:
            if (fields != fields0):
                print('All training files must have same fields for training')
                sys.exit(-1)
        
        if(data_usage<1):    #Reduce data by picking random selection of data, keep it sorted to prevent mixing the training and validation dataset 
            n = int(len(d)*data_usage)
            d_ix = np.sort(np.random.choice(d.shape[0],n,replace=False))
        else:
            d_ix = np.array(range(len(d)))
        
        if valid_split>0:
            n_valid = int(len(d_ix)*valid_split)    #Number of samples for validation
            v_ix = d_ix[:n_valid]
            d_ix = d_ix[n_valid:]
            v_labels.append(d[v_ix,fields.index('Classification')])         
            v_data.append(d[v_ix,(fields.index('Classification')+1):])  #Only use fields after classification field for training (X,Y,Z, beamnumber, pingnumber are irrelevant)

        labels.append(d[d_ix,fields.index('Classification')])         
        data.append(d[d_ix,(fields.index('Classification')+1):])  #Only use fields after classification field for training (X,Y,Z, beamnumber, pingnumber are irrelevant)

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    
    if valid_split>0:
        v_data = np.concatenate(v_data)
        v_labels = np.concatenate(v_labels)
        print("Training   data %d data points %d parameters"%(data.shape)) 
        print("Validation data %d data points %d parameters"%(v_data.shape)) 
        return data,labels,v_data,v_labels
    else:
        print("Training data %d data points %d parameters"%(data.shape)) 
        return data,labels


if has_train_data:
    train_data,train_labels,validation_data, validation_labels = read_super_batch_from_files(args.train,valid_split = 0.25)
    #if True: #Shuffle data and lables together
    #    ix = np.array(range(len(data)))
    #    np.random.shuffle(ix)
    #    data = data[ix,:]
    #    labels = labels[ix]


    #Labels are one hot coded, so a SAMPLESxCLASSES array of ones and zeros
    train_labels = to_categorical(train_labels, num_classes=n_classes) 
    validation_labels = to_categorical(validation_labels, num_classes=n_classes) 
    
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=8*1024, validation_data=(validation_data, validation_labels))


if has_test_data:
    print("Reading test data from %s" % (args.test[0]))
    f_test = CcSbf(args.test[0])
    test_fields,test_data,test_offset = f_test.read_raw()
    test_labels = test_data[:,test_fields.index('Classification')]         
    test_data_sel = test_data[:,(test_fields.index('Classification')+1):]  #Only use fields after classification field for training (X,Y,Z, beamnumber, pingnumber are irrelevant)
    #Limit test data to 2M samples
    test_labels = to_categorical(test_labels[:max_test_data], num_classes=n_classes)
    predictions = model(test_data_sel[:max_test_data])
    test_labels_predicted = np.array(predictions)

if write_model:
    print("Writing keras model to %s" % (args.model_out[0]))
    model.save(args.model_out[0])

from matplotlib.pylab import *
if has_train_data:
    ax1=plt.subplot(2,2,1)
    loss = history.history["loss"] 
    val_loss = history.history["val_loss"] 
    epochs = range(1, len(loss) + 1) 
    ax1.plot(epochs, loss, "bo", label="Training loss") 
    ax1.plot(epochs, val_loss, "b", label="Validation loss") 
    ax1.set_title("Training and validation loss") 
    #ax1.set_xlabel("Epochs") 
    ax1.set_ylabel("Loss") 
    ax1.legend() 

    ax2=plt.subplot(2,2,2)
    acc = history.history["accuracy"] 
    val_acc = history.history["val_accuracy"] 
    ax2.plot(epochs, acc, "bo", label="Training accuracy") 
    ax2.plot(epochs, val_acc, "b", label="Validation accuracy") 
    ax2.set_title("Training and validation accuracy") 
    ax2.set_xlabel("Epochs") 
    ax2.set_ylabel("Accuracy") 
    ax2.legend() 

if has_test_data:
    ax3=plt.subplot(2,2,3)
    ax3.set_title("Test data labels") 
    ax3.imshow(test_labels[:max_test_data],aspect='auto')
    ax3.set_ylabel("Samples") 
    ax3.set_xlabel("Label") 
    ax4=plt.subplot(2,2,4)
    ax4.set_title("Test data predicted labels") 
    ax4.imshow(test_labels_predicted,aspect='auto')
    ax4.set_xlabel("Label") 

plt.show()

if has_test_data:

    #Open up file with original full (non-augmented) data

    #Drop neighborhood data fields
    gix = np.where(['neigh' not in  x for x in test_fields])[0]
    test_fields =  [test_fields[x] for x in gix]
    test_data = test_data[:,gix]


    for c in range(test_labels_predicted.shape[1]):
        test_fields.append('predicted_class_%d_prob'%(c))                 #Append class names for class probability data

    if (len(test_data)>max_test_data):
        test_labels_predicted_padded = np.zeros((test_data.shape[0], test_labels_predicted.shape[1]))
        test_labels_predicted_padded[:max_test_data] = test_labels_predicted
    else:
        test_labels_predicted_padded = test_labels_predicted

    test_data = np.concatenate([test_data,test_labels_predicted_padded],axis=1)      #Append class probability data

    test_fields.append('predicted_class')                                #Append class name for prediced class
    test_data = np.concatenate([test_data,np.array([np.argmax(test_labels_predicted_padded,axis=1)]).T],axis=1)
    #Write back test data set with classification data
    f_test_out_fname = args.test[0][:args.test[0].find('.')]+'_classified.sbf'
    f_test_out = CcSbf(f_test_out_fname)
    f_test_out.write_raw(test_fields,test_data,test_offset)




