import numpy as np
import os
import collections
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.python.keras.backend import mean

# Useful Constants
DATASET_PATH = os.path.join('datasets', 'UCI HAR Dataset')
TRAIN = os.path.join('train')
TEST = os.path.join('test')

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    y_ = y_ - 1
    return np.array([y__[0] for y__ in y_])

'''def normalization_perChannel(X_train, X_test):
    print('Normalization per channel')
    for i in range(X_train.shape[2]):
        scaler = MinMaxScaler()

        train_data = X_train[:,:,i]
        X_train[:,:,i] = scaler.fit_transform(train_data)

        test_data = X_test[:,:,i]
        X_test[:,:,i] = scaler.transform(test_data)

        del scaler

    return X_train, X_test'''

def normalization_perChannel2(X_train, X_test):
    print('Normalization per channel')

    print(X_train.shape)
    oo=input()

    [shape0_tr, shape1_tr, shape2_tr] = X_train.shape
    x_train = X_train.reshape(shape0_tr*shape1_tr, shape2_tr)
    
    [shape0_te, shape1_te, shape2_te] = X_test.shape
    x_test = X_test.reshape(shape0_te*shape1_te, shape2_te)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    print(x_train.shape)
    oo=input()

    mean_train = np.mean(x_train, axis=0)
    std_train = np.std(x_train, axis=0)
    x_train = (x_train - mean_train) / std_train
    x_test = (x_test - mean_train) / std_train

    X_train = x_train.reshape(shape0_tr, shape1_tr, shape2_tr)
    X_test = x_test.reshape(shape0_te, shape1_te, shape2_te)

    return X_train, X_test

def normalization_perChannel(X_train, X_test):
    print('Normalization per channel')

    for i in range(X_train.shape[2]):
        x_train = X_train[:,:,i]
        [shape0_tr, shape1_tr] = x_train.shape
        x_train = x_train.reshape(shape0_tr*shape1_tr)
        
        x_test = X_test[:,:,i]
        [shape0_te, shape1_te] = x_test.shape
        x_test = x_test.reshape(shape0_te*shape1_te)

        mean_train = np.min(x_train, axis=0)
        std_train = np.max(x_train, axis=0)
        x_train = (x_train - mean_train) / (std_train-mean_train)
        x_test = (x_test - mean_train) / (std_train-mean_train)

        X_train[:,:,i] =  x_train.reshape(shape0_tr,shape1_tr)
        X_test[:,:,i] = x_test.reshape(shape0_te,shape1_te)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, X_test

def load_UCI_dataset(flatten):
    print('loading UCI HAR dataset..')
    print('labels: ', LABELS)
    
    X_train_signals_paths = [ os.path.join(DATASET_PATH, TRAIN, "Inertial Signals", signal + "train.txt") for signal in INPUT_SIGNAL_TYPES]
    X_test_signals_paths = [ os.path.join(DATASET_PATH, TEST, "Inertial Signals", signal + "test.txt") for signal in INPUT_SIGNAL_TYPES]

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    # rescaling using per-channel normalization
    X_train, X_test = normalization_perChannel(X_train, X_test)

    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
    else:
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1, X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1, X_test.shape[2]))

    y_train_path = os.path.join(DATASET_PATH, TRAIN, "y_train.txt")
    y_test_path = os.path.join(DATASET_PATH, TEST, "y_test.txt")

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    print("X train (shape): ", X_train.shape)
    print("X test (shape): ", X_test.shape)

    print(collections.Counter(y_train))
    print(collections.Counter(y_test))

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_train, y_train, random_state=0)

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, X_test, y_train, y_test, LABELS

def prepare_for_DSC(X_train, X_test):

    X_train_copy = np.array(X_train, copy=True)  
    inverted_X_train = X_train[:,:64,:,:]
    inverted_X_train = inverted_X_train[:,::-1,:,:]
    future_X_train = X_train[:,64:,:,:]
    X_train = X_train_copy[:,:64,:,:]

    X_train_copy = np.array(X_test, copy=True)  
    inverted_X_test = X_test[:,:64,:,:]
    inverted_X_test = inverted_X_test[:,::-1,:,:]
    future_X_test= X_test[:,64:,:,:]
    X_test = X_train_copy[:,:64,:,:]

    X_train = X_train[:,:,0,:]
    inverted_X_train = inverted_X_train[:,:,0,:]
    future_X_train = future_X_train[:,:,0,:]

    X_test = X_test[:,:,0,:]
    inverted_X_test = inverted_X_test[:,:,0,:]
    future_X_test = future_X_test[:,:,0,:]

    return X_train, inverted_X_train, future_X_train, X_test, inverted_X_test, future_X_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, LABELS = load_UCI_dataset(flatten=False)
    X_train, inverted_X_train, future_X_train, X_test, inverted_X_test, future_X_test = prepare_for_DSC(X_train, X_test)
