import numpy as np
import pandas as pd
from paretoset import paretoset
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class Server:
    
    def __init__(self, client_model_1, client_model_2):
        self.client_model_1 = client_model_1
        self.client_model_2 = client_model_2
        self.model_1 = client_model_1.get_params()
        self.model_2 = client_model_2.get_params()
        self.selected_clients = []
        self.updates_1 = []
        self.updates_2 = []
        self.gamma = 0.9
        self.L1 = {}
        self.U1 = {}
        self.N1 = {}
        self.L2 = {}
        self.U2 = {}
        self.N2 = {}
        self.T = 0
        self.p1 = {}
        self.p2 = {}
        self.A1 = {}
        self.A2 = {}
        self.A1_prev = {}
        self.A2_prev = {}
        self.model_nos = {}

    def initialize_UCB_params(self, possible_clients):
        for c in possible_clients:
            self.L1[c] = 0
            self.U1[c] = 0
            self.N1[c] = 0
            self.L2[c] = 0
            self.U2[c] = 0
            self.N2[c] = 0
            self.A1[c] = 0
            self.A2[c] = 0
            self.model_nos[c] = 0

    def policy0(self,possible_clients, num_clients):
        selected_clients = np.random.choice(possible_clients, num_clients)
        model_nos = {}
        for c in possible_clients:
            model_nos[c] = np.random.choice([1,2])

        return selected_clients.copy(), model_nos.copy()

    def policy1(self,possible_clients, num_clients):
        A = {}
        model_nos = {}
        for c in possible_clients:
            model_nos[c] = 1 if self.A1[c] > self.A2[c] else 2

            A[c] = self.p1[c.id]*self.A1[c] + self.p2[c.id]*self.A2[c]

        sorted_A = sorted(A.items(), key = lambda kv: kv[1], reverse = True)
        selected_clients = []
        for t in sorted_A[0:num_clients]:
            selected_clients.append(t[0])

        return selected_clients.copy(), model_nos.copy()

    def policy1a(self,possible_clients, num_clients):
        A = {}
        model_nos = {}
        for c in possible_clients:
            model_nos[c] = 1 if self.U1[c] > self.U2[c] else 2

            A[c] = self.p1[c.id]*self.A1[c] + self.p2[c.id]*self.A2[c]

        sorted_A = sorted(A.items(), key = lambda kv: kv[1], reverse = True)
        selected_clients = []
        for t in sorted_A[0:num_clients]:
            selected_clients.append(t[0])

        return selected_clients.copy(), model_nos.copy()

    def policy2(self, possible_clients, num_clients):
        num_non_dom_clients = {}
        model_nos = self.model_nos.copy()

        A_df = pd.DataFrame({'A1':list(self.A1.values()), 'A2':list(self.A2.values())})
        mask = paretoset(A_df, sense=["max", "max"])
        pareto_df = A_df[mask]

        for c in possible_clients:
            model_nos[c] = 1 if self.U1[c] > self.U2[c] else 2
        #     if not c in num_non_dom_clients.keys():
        #         num_non_dom_clients[c] = 0
        #     for d in possible_clients:
        #         if sum(np.asarray([self.A1[c], self.A2[c]]) > np.asarray([self.A1[d], self.A2[d]])) > 0:
        #             num_non_dom_clients[c] += 1

        pareto_optimal_clients = [possible_clients[i] for i in pareto_df.index]
        # for c in possible_clients:
        #     if num_non_dom_clients[c] == len(possible_clients) - 1: pareto_optimal_clients.append(c)

        selected_clients = []
        if len(pareto_optimal_clients) > num_clients:
            selected_clients = np.random.choice(pareto_optimal_clients, num_clients)
        else:
            selected_clients = pareto_optimal_clients

        return selected_clients.copy(), model_nos.copy()

    def policy3(self, possible_clients, num_clients):
        sorted_A1 = sorted(self.A1.items(), key = lambda kv: kv[1], reverse = True)
        sorted_A2 = sorted(self.A2.items(), key = lambda kv: kv[1], reverse = True)

        chance = 1
        i = 0
        j = 0
        model_nos = self.model_nos.copy()
        selected_clients = []
        while(len(selected_clients) < num_clients):
            if chance == 1:
                c = sorted_A1[i][0]
                if c in selected_clients:
                    i = i+1
                    model_nos[c] = 1 if self.U1[c] > self.U2[c] else 2
                else:
                    selected_clients.append(c)
                    model_nos[c] = 1
                    i = i+1
                    chance = 2
            else:
                c = sorted_A2[j][0]
                if c in selected_clients:
                    j = j+1
                    model_nos[c] = 1 if self.U1[c] > self.U2[c] else 2
                else:
                    selected_clients.append(c)
                    model_nos[c] = 2
                    j = j+1
                    chance = 1

        return selected_clients, model_nos


    def policy3a(self, possible_clients, num_clients):
        sorted_A1 = sorted(self.A1.items(), key = lambda kv: kv[1], reverse = True)
        sorted_A2 = sorted(self.A2.items(), key = lambda kv: kv[1], reverse = True)

        chance = 1
        i = 0
        j = 0
        model_nos = self.model_nos.copy()
        selected_clients = []
        while(len(selected_clients) < num_clients):
            if chance == 1:
                c = sorted_A1[i][0]
                if c in selected_clients:
                    i = i+1
                    #if model_nos[c] = 1 if self.U1[c] > self.U2[c] else 2
                else:
                    selected_clients.append(c)
                    model_nos[c] = 1
                    i = i+1
                    chance = 2
            else:
                c = sorted_A2[j][0]
                if c in selected_clients:
                    j = j+1
                    #model_nos[c] = 1 if self.U1[c] > self.U2[c] else 2
                else:
                    selected_clients.append(c)
                    model_nos[c] = 2
                    j = j+1
                    chance = 1

        return selected_clients, model_nos


    def policy4(self, possible_clients, num_clients):
        num_non_dom_clients = {}
        model_nos = self.model_nos.copy()
        
        A_df = pd.DataFrame({'A1':list(self.A1.values()), 'A2':list(self.A2.values())})
        mask = paretoset(A_df, sense=["max", "max"])
        pareto_df = A_df[mask]

        pareto_optimal_clients = [possible_clients[i] for i in pareto_df.index]

        sorted_A1 = sorted(self.A1.items(), key = lambda kv: kv[1], reverse = True)
        sorted_A2 = sorted(self.A2.items(), key = lambda kv: kv[1], reverse = True)

        A1_ranklist = []
        for t in sorted_A1:
            A1_ranklist.append(t[0])

        A2_ranklist = []
        for t in sorted_A2:
            A2_ranklist.append(t[0])

        for c in pareto_optimal_clients:
            rank_1 = A1_ranklist.index(c)
            rank_2 = A2_ranklist.index(c)
            model_nos[c] = 1 if rank_1 < rank_2 else 2

        selected_clients = []
        if len(pareto_optimal_clients) > num_clients:
            selected_clients = np.random.choice(pareto_optimal_clients, num_clients)
        else:
            selected_clients = pareto_optimal_clients

        return selected_clients.copy(), model_nos.copy()


    def policy2a(self, possible_clients, num_clients, relaxation):
        num_non_dom_clients = {}
        model_nos = {}
        for c in possible_clients:
            model_nos[c] = 1 if self.U1[c] > self.U2[c] else 2
            num_non_dom_clients[c] = 0
            for d in possible_clients:
                if sum(np.asarray([self.A1[c], self.A2[c]]) > np.asarray([self.A1[d], self.A2[d]])) > 0:
                    num_non_dom_clients[c] += 1

        pareto_optimal_clients = []
        for c in possible_clients:
            if num_non_dom_clients[c] >= len(possible_clients) - 1 - relaxation: pareto_optimal_clients.append(c)

        selected_clients = []
        if len(pareto_optimal_clients) > num_clients:
            selected_clients = np.random.choice(pareto_optimal_clients, num_clients)
        else:
            selected_clients = pareto_optimal_clients

        return selected_clients.copy(), model_nos.copy()





    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        if my_round == 0:
            self.initialize_UCB_params(possible_clients)

            _,_, num_samples_1, num_samples_2 = self.get_clients_info(possible_clients)
            self.p1 = num_samples_1.copy()
            self.p2 = num_samples_2.copy()
            p1_sum = sum(self.p1.values())
            p2_sum = sum(self.p2.values())
            for c in self.p1.keys():
                self.p1[c] = self.p1[c]/p1_sum
                self.p2[c] = self.p2[c]/p2_sum

            self.selected_clients = possible_clients
            for c in self.selected_clients:
                self.model_nos[c] = 1

        if my_round == 1:
            for c in possible_clients:
                self.L1[c] = c.loss_history_1[-1] + self.gamma*self.L1[c]
                self.N1[c] = int(c.loss_history_1[-1]>0) + self.gamma*self.N1[c]
                self.T = 1 + self.gamma*self.T

            self.selected_clients = possible_clients
            for c in self.selected_clients:
                self.model_nos[c] = 2

        num_clients = min(num_clients, len(possible_clients))
        #np.random.seed(my_round)
        if my_round > 1:
            for c in possible_clients:
                self.L1[c] = c.loss_history_1[-1] + self.gamma*self.L1[c]
                self.L2[c] = c.loss_history_2[-1] + self.gamma*self.L2[c]
                self.N1[c] = int(c.loss_history_1[-1]>0) + self.gamma*self.N1[c]
                self.N2[c] = int(c.loss_history_2[-1]>0) + self.gamma*self.N2[c]
                self.T = 1 + self.gamma*self.T
                self.U1[c] = np.sqrt(2*np.log(self.T)/self.N1[c])
                self.U2[c] = np.sqrt(2*np.log(self.T)/self.N2[c])
                self.A1_prev = self.A1.copy()
                self.A2_prev = self.A2.copy()
                self.A1[c] = (self.L1[c]/self.N1[c]) + self.U1[c]
                self.A2[c] = (self.L2[c]/self.N2[c]) + self.U2[c]

            #print(possible_clients)
            #self.selected_clients, self.model_nos = self.policy1(possible_clients, num_clients)
            self.selected_clients, self.model_nos = self.policy3a(possible_clients, num_clients)

            #print(len(self.selected_clients))

        for c in possible_clients:
            c.loss_history_1.append(0)
            c.loss_history_2.append(0)

        #return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        # sys_metrics = {
        #     c.id: {BYTES_WRITTEN_KEY: 0,
        #            BYTES_READ_KEY: 0,
        #            LOCAL_COMPUTATIONS_KEY: 0} for c in clients}

        #model_nos = np.random.choice(a = [1,2], size = (len(clients),1))

        for c in clients:
            if self.model_nos[c]==1:
                c.model_1.set_params(self.model_1)
                comp, num_samples, update, loss = c.train(1, num_epochs, batch_size, minibatch)

                # sys_metrics[c.id][BYTES_READ_KEY] += c.model_1.size
                # sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model_1.size
                # sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

                self.updates_1.append((num_samples, update))
                c.loss_history_1[-1] = loss
            else:
                c.model_2.set_params(self.model_2)
                comp, num_samples, update, loss = c.train(2, num_epochs, batch_size, minibatch)

                # sys_metrics[c.id][BYTES_READ_KEY] += c.model_2.size
                # sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model_2.size
                # sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

                self.updates_2.append((num_samples, update))
                c.loss_history_2[-1] = loss

        print("No. of Model 1 clients: ", len(self.updates_1), " and No. of Model 2 clients: ", len(self.updates_2))
        #return sys_metrics

    def update_model(self):
        if len(self.updates_1) > 0:
            total_weight = 0.
            base = [0] * len(self.updates_1[0][1])
            for (client_samples, client_model) in self.updates_1:
                total_weight += client_samples
                for i, v in enumerate(client_model):
                    base[i] += (client_samples * v.astype(np.float64))
            averaged_soln = [v / total_weight for v in base]

            self.model_1 = averaged_soln
            self.updates_1 = []
        
        if len(self.updates_2) > 0:
            total_weight = 0.
            base = [0] * len(self.updates_2[0][1])
            for (client_samples, client_model) in self.updates_2:
                total_weight += client_samples
                for i, v in enumerate(client_model):
                    base[i] += (client_samples * v.astype(np.float64))
            averaged_soln = [v / total_weight for v in base]

            self.model_2 = averaged_soln
            self.updates_2 = []

    def test_model(self, model_no, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            if model_no==1:
                client.model_1.set_params(self.model_1)
                c_metrics = client.test(model_no, set_to_use)
                metrics[client.id] = c_metrics
            else:
                client.model_2.set_params(self.model_2)
                c_metrics = client.test(model_no, set_to_use)
                metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples_1 = {c.id: c.num_samples_1 for c in clients}
        num_samples_2 = {c.id: c.num_samples_2 for c in clients}
        return ids, groups, num_samples_1, num_samples_2

    def save_model_1(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model_1.set_params(self.model_1)
        model_sess =  self.client_model_1.sess
        return self.client_model_1.saver.save(model_sess, path)

    def save_model_2(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model_2.set_params(self.model_2)
        model_sess =  self.client_model_2.sess
        return self.client_model_2.saver.save(model_sess, path)

    def close_model(self):
        self.client_model_1.close()
        self.client_model_2.close()