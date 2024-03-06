#Load training dataset
from pickle import load
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np


input_filename = 'survey_dataset1'

with open(input_filename+'_augmented.pickle','rb') as f:
    data = load(f)

test_data_range = [0,50000]
validation_data_range = [50000,100000]
train_data_range = [100000,-1]


raw_test_data = data['data'][test_data_range[0]:test_data_range[1]]
raw_test_labels = data['labels'][test_data_range[0]:test_data_range[1]]
raw_validation_data = data['data'][validation_data_range[0]:validation_data_range[1]]
raw_validation_labels = data['labels'][validation_data_range[0]:validation_data_range[1]]
raw_train_data = data['data'][train_data_range[0]:train_data_range[1]]
raw_train_labels = data['labels'][train_data_range[0]:train_data_range[1]]

#Data needs no reformating
train_data = raw_train_data
test_data = raw_test_data
validation_data = raw_validation_data
#Labels are one hot coded, so a SAMPLESxCLASSES array of ones and zeros
train_labels = to_categorical(raw_train_labels, num_classes=5) 
test_labels = to_categorical(raw_test_labels, num_classes=5)
validation_labels = to_categorical(raw_validation_labels, num_classes=5)


if False: #Processs the rauters dataset (Textbook example)
    from tensorflow.keras.datasets import reuters 
    (raw_train_data, raw_train_labels), (raw_test_data, raw_test_labels) = reuters.load_data( num_words=10000)
    #raw_train_data is an array of 8982 samples, each a lists containing an arbitrary number of words coded as integers [0-9999]
    #raw_train_labels is an array of 8982 integers [0-45], target class of each sample

    #raw_test_data is similar to raw_train_data, but only 2246 samples
    #raw_test_labels is similar to raw_train_labels, but only 2246 values


    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension)) 
        for i, sequence in enumerate(sequences): 
            for j in sequence:
                results[i, j] = 1. 
        return results

    ###Format data correctly for processing
    #Data is one-hot coded, so a SAMPLESxWORDS array of ones and zeros
    train_data = vectorize_sequences(raw_train_data) 
    test_data = vectorize_sequences(raw_test_data)
    #Labels are one hot coded, so a SAMPLESxCLASSES array of ones and zeros
    train_labels = to_categorical(raw_train_labels) 
    test_labels = to_categorical(raw_test_labels)

    if False:       #Split train data into train and validation data 
        validation_data = train_data[:1000] 
        train_data = train_data[1000:] 
        validation_labels = train_labels[:1000] 
        train_labels = train_labels[1000:]
    else:           #Use test data as validation data
        validation_data = test_data
        validation_labels = test_labels

#Set up NN model

n_train_samples = train_labels.shape[0]
n_classes = train_labels.shape[1]

model = keras.Sequential([
        layers.Dense(64,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(n_classes,activation='softmax')
        ])

model.compile(  optimizer=keras.optimizers.RMSprop(learning_rate=0.00003),
                loss='categorical_crossentropy',
                metrics=['accuracy'])



history = model.fit(train_data, train_labels, epochs=15, batch_size=1024, validation_data=(validation_data, validation_labels))

predictions = model.predict(test_data)
test_labels_predicted = np.array(predictions)


from matplotlib.pylab import *
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

ax3=plt.subplot(2,2,3)
ax3.set_title("Test data labels") 
ax3.imshow(test_labels,aspect='auto')
ax3.set_ylabel("Samples") 
ax3.set_xlabel("Label") 
ax4=plt.subplot(2,2,4)
ax4.set_title("Test data predicted labels") 
ax4.imshow(test_labels_predicted,aspect='auto')
ax4.set_xlabel("Label") 

plt.show()

#Open up file with original full (non-augmented) data
from py_cc_sbf import CcSbf
f_in = CcSbf(input_filename+'.sbf')
fields,data,offset = f_in.read_raw()
data = data[test_data_range[0]:test_data_range[1],:]  #Extract test data section

for c in range(test_labels_predicted.shape[1]):
    fields.append('predicted_class_%d_prob'%(c))                 #Append class names for class probability data
data = np.concatenate([data,test_labels_predicted],axis=1)      #Append class probability data

fields.append('predicted_class')                                #Append class name for prediced class
data = np.concatenate([data,np.array([np.argmax(test_labels_predicted,axis=1)]).T],axis=1)
#Write back test data set with classification data
f_out = CcSbf(input_filename+'_classified.sbf')
f_out.write_raw(fields,data,offset)




