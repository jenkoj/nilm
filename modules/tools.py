import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools 

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense



def my_print(s):
    print(s)
    with open('log.log', 'a') as f:
        f.write(str(s) + '\n')


def plot_confusion_matrix_norm(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(10,10))

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = np.around(cm, decimals=2)

        cm[np.isnan(cm)] = 0.0

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
    plt.colorbar()
    
    #plt.savefig(("out/B"+dataset.split("_")[0]+"TL"+datasetTL+seed),dpi=300)

#fetch array of appliances
def get_data(file):
    
    enc_appliances = np.array(file["appliances/classes"])
    appliances = [n.decode("utf-8") for n in enc_appliances]
    num_of_classes = file["appliances/classes"].shape[0]
    
    print(appliances)

    x_test = np.array(file['data/test']['gaf'])
    #x_test = 0
    y_test = np.array(file['labels/test']['gaf'])
    y_train = file["labels/train/gaf"][:]
    class_weights = class_weight.compute_class_weight(class_weight="balanced", classes = np.unique(y_train), y=y_train)
    d_class_weights = dict(enumerate(class_weights))
    print(d_class_weights)
    
    print(" ")
    print("Tests ")
    for ctest,ctrain,appl in zip(np.unique(y_test,return_counts=True)[1],np.unique(y_train,return_counts=True)[1],appliances):
        print(appl,"test:",ctest,"train:",ctrain)
    
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_of_classes)
    
    return x_test, y_test, d_class_weights, num_of_classes, appliances

def get_data_split(file,seed):

    #train test split
    print("...spliting")

    print("loading labels")
    try:
        Y = file["labels"][:]
    except:
        Y = file["labels/gaf"][:]
    print("loading images")


    x_train, x_test, y_train, y_test = train_test_split(file["data/gaf"][:],Y,test_size=0.2,random_state=int(seed),stratify=Y)
    print("split shapes shapes:")
    
    print(len(x_train))
    print(len(x_test))

    print("compute class weights")
    class_weights = class_weight.compute_class_weight(class_weight="balanced", classes = np.unique(y_train), y=y_train)
    d_class_weights = dict(enumerate(class_weights))
    print(d_class_weights)


    enc_appliances = np.array(file["appliances/classes"])
    appliances = [n.decode("utf-8") for n in enc_appliances]
    num_of_classes = file["appliances/classes"].shape[0]

    print(" ")
    print("Tests ")
    for ctest,ctrain,appl in zip(np.unique(y_test,return_counts=True)[1],np.unique(y_train,return_counts=True)[1],appliances):
        print(appl,"test:",ctest,"train:",ctrain)
    
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_of_classes)
    
    return x_test,y_test,d_class_weights,num_of_classes,appliances

def evaluate_model(model_used,x_test,y_test,appliances):
    #Print results and plot confusion matrix

    Y_pred = model_used.predict(x_test, verbose = 2)
    y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(y_test, axis=-1)
    C = confusion_matrix(Y_test, y_pred)
    #print(confusion_matrix(Y_test, y_pred))
    #recision,recall,fscore,support=score(Y_test, y_pred,average='macro')
    #print("F1 SCORE",fscore)
    plot_confusion_matrix_norm(C, appliances, normalize=True)
    print(classification_report(Y_test, y_pred, target_names=appliances))


def create_tl_model(model,type,model_seed,path,num_of_classes):
    
    print("compling TL model..")
    #create TL model 
    path_model = path+"/models/"+type+"/"+model+"_"+str(model_seed)
    #path_model = path+"/models/refit/GSAF/LSTM3DV56/model2"

    model = keras.models.load_model(path_model)
    #create trasfer learning model 
    model.trainable = False
    base_output = model.layers[-2].output
    hidden4 = Dense(64, activation='relu')(base_output)
    hidden3 = Dense(32, activation='relu')(hidden4)
    hidden2 = Dense(16, activation='relu')(hidden3)
    hidden = Dense(num_of_classes, activation='softmax')(hidden2)

    model2 = keras.models.Model(model.inputs, hidden)
    lr = 0.002
    adam = optimizers.Adam(lr = lr)
    model2.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model2