import random
import warnings


class Client:
    
    def __init__(self, client_id, group=None, train_data_1={'x' : [],'y' : []}, eval_data_1={'x' : [],'y' : []}, model_1=None, train_data_2={'x' : [],'y' : []}, eval_data_2={'x' : [],'y' : []}, model_2=None):
        self._model_1 = model_1
        self._model_2 = model_2
        self.id = client_id
        self.group = group
        self.train_data_1 = train_data_1
        self.eval_data_1 = eval_data_1
        self.train_data_2 = train_data_2
        self.eval_data_2 = eval_data_2
        self.loss_history_1 = []
        self.loss_history_2 = []

    def train(self, model_no, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        if model_no == 1:
            if minibatch is None:
                data = self.train_data_1
                comp, update, loss = self.model_1.train(data, num_epochs, batch_size)
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data_1["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data_1["x"], self.train_data_1["y"])), num_data))
                data = {'x': xs, 'y': ys}

                # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                num_epochs = 1
                comp, update, loss = self.model_1.train(data, num_epochs, num_data)
        else:
            if minibatch is None:
                data = self.train_data_2
                comp, update, loss = self.model_2.train(data, num_epochs, batch_size)
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data_2["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data_2["x"], self.train_data_2["y"])), num_data))
                data = {'x': xs, 'y': ys}

                # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                num_epochs = 1
                comp, update, loss = self.model_2.train(data, num_epochs, num_data)

        num_train_samples = len(data['y'])
        return comp, num_train_samples, update, loss

    def test(self, model_no, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if model_no==1:
            if set_to_use == 'train':
                data = self.train_data_1
            elif set_to_use == 'test' or set_to_use == 'val':
                data = self.eval_data_1
            return self.model_1.test(data)
        else:
            if set_to_use == 'train':
                data = self.train_data_2
            elif set_to_use == 'test' or set_to_use == 'val':
                data = self.eval_data_2
            return self.model_2.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data_1['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data_1['y'])

    @property
    def num_samples_1(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data_1 is not None:
            train_size = len(self.train_data_1['y'])

        test_size = 0 
        if self.eval_data_1 is not  None:
            test_size = len(self.eval_data_1['y'])
        return train_size + test_size

    @property
    def num_samples_2(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data_2 is not None:
            train_size = len(self.train_data_2['y'])

        test_size = 0 
        if self.eval_data_2 is not  None:
            test_size = len(self.eval_data_2['y'])
        return train_size + test_size

    @property
    def model_1(self):
        """Returns this client reference to model being trained"""
        return self._model_1

    @model_1.setter
    def model_1(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model_1 = model_1

    @property
    def model_2(self):
        """Returns this client reference to model being trained"""
        return self._model_2

    @model_2.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model_2 = model_2
