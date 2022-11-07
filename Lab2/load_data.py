

# load cifar-10 data

def load_data(dir="./batched_cifar10"):
    import pickle
    import numpy as np

    X_train = []
    Y_train = []

    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        X_train.append(data_dict[b'data'])
        Y_train += data_dict[b'labels']

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.array(Y_train)

    with open(dir + r'/test_batch', 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    X_test = np.array(data_dict[b'data'])
    Y_test = np.array(data_dict[b'labels'])

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
