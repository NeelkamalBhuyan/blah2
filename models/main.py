"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf
import pandas as pd

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

#df_main = pd.DataFrame(columns = ['round', 'model 1 train acc', 'model 1 train loss', 'model 2 train acc', 'model 2 train loss', 'model 1 test acc', 'model 1 test loss', 'model 2 test acc', 'model 2 test loss'])

def main():

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)


    model_path_1 = '%s/%s.py' % (args.dataset_1, args.model_1)
    if not os.path.exists(model_path_1):
        print('Please specify a valid dataset and a valid model.')
    model_path_1 = '%s.%s' % (args.dataset_1, args.model_1)

    model_path_2 = '%s/%s.py' % (args.dataset_2, args.model_2)
    if not os.path.exists(model_path_2):
        print('Please specify a valid dataset and a valid model.')
    model_path_2 = '%s.%s' % (args.dataset_2, args.model_2)
    


    print('############################## %s ##############################' % model_path_1)
    print('############################## %s ##############################' % model_path_2)


    mod_1 = importlib.import_module(model_path_1)
    ClientModel_1 = getattr(mod_1, 'ClientModel')

    mod_2 = importlib.import_module(model_path_2)
    ClientModel_2 = getattr(mod_2, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset_2][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params_1 = MODEL_PARAMS[model_path_1]
    if args.lr != -1:
        model_params_list = list(model_params_1)
        model_params_list[0] = args.lr
        model_params_1 = tuple(model_params_list)

    model_params_2 = MODEL_PARAMS[model_path_2]
    if args.lr != -1:
        model_params_list = list(model_params_2)
        model_params_list[0] = args.lr
        model_params_2 = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model_1 = ClientModel_1(args.seed, *model_params_1)
    client_model_2 = ClientModel_2(args.seed, *model_params_2)


    # Create server
    server = Server(client_model_1, client_model_2)




    # Create clients
    clients = setup_clients(args.dataset_1, client_model_1, args.dataset_2, client_model_2, args.use_val_set)
    client_ids, client_groups, client_num_samples_1, client_num_samples_2 = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # metrics df
    df_main = []

    # Initial status
    print('--- Random Initialization ---')
    #stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    #sys_writer_fn = get_sys_writer_function(args)
    all_metrics = print_stats(0, server, clients, client_num_samples_1, client_num_samples_2, args, args.use_val_set)
    df_main.append([0]+all_metrics)

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples_1, c_num_samples_2 = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        #sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)
        
        # Update server model
        server.update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            # collect metrics
            all_metrics = print_stats(i + 1, server, clients, client_num_samples_1, client_num_samples_2, args, args.use_val_set)
            df_main.append([i+1]+all_metrics)
    
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset_1)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model_1(os.path.join(ckpt_path, '{}.ckpt'.format(args.model_1)))
    print('Model saved in path: %s' % save_path)

    ckpt_path = os.path.join('checkpoints', args.dataset_2)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model_2(os.path.join(ckpt_path, '{}.ckpt'.format(args.model_2)))
    print('Model saved in path: %s' % save_path)

    # Close models
    server.close_model()

    metrics_df = pd.DataFrame(df_main, columns = ['round', 'model 1 train acc', 'model 1 train loss', 'model 2 train acc', 'model 2 train loss', 'model 1 test acc', 'model 1 test loss', 'model 2 test acc', 'model 2 test loss'])
    metrics_df.to_csv("metrics/my_metrics.csv")

    client_loss_history = {}
    for c in clients:
        key = str(c.id) + '_ model_1'
        client_loss_history[key] = c.loss_history_1
        key = str(c.id) + '_ model_2'
        client_loss_history[key] = c.loss_history_2

    client_loss_history_df = pd.DataFrame(client_loss_history)
    client_loss_history_df.to_csv("metrics/client_loss_history.csv")


def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users_1, groups_1, train_data_1, test_data_1, model_1, users_2, groups_2, train_data_2, test_data_2, model_2):
    if len(groups_1) == 0:
        groups_1 = [[] for _ in users_1]

    if len(groups_2) == 0:
        groups_2 = [[] for _ in users_2]

    clients = [Client(u1, g1, train_data_1[u1], test_data_1[u1], model_1, train_data_2[u2], test_data_2[u2], model_2) for u1, g1, u2, g2 in zip(users_1, groups_1, users_2, groups_2)]
    return clients


def setup_clients(dataset_1, model_1, dataset_2, model_2, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset_1, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset_1, 'data', eval_set)

    users_1, groups_1, train_data_1, test_data_1 = read_data(train_data_dir, test_data_dir)

    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset_2, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset_2, 'data', eval_set)

    users_2, groups_2, train_data_2, test_data_2 = read_data(train_data_dir, test_data_dir)


    clients = create_clients(users_1, groups_1, train_data_1, test_data_1, model_1, users_2, groups_2, train_data_2, test_data_2, model_2)

    return clients


# def get_stat_writer_function(ids, groups, num_samples, args):

#     def writer_fn(num_round, metrics, partition):
#         metrics_writer.print_metrics(
#             num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

#     return writer_fn


# def get_sys_writer_function(args):

#     def writer_fn(num_round, ids, metrics, groups, num_samples):
#         metrics_writer.print_metrics(
#             num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

#     return writer_fn


def print_stats(
    num_round, server, clients, num_samples_1, num_samples_2, args, use_val_set):
    
    # train and test data accuracy and loss (2*2 = 4) for each model (total 8 metrics)
    all_metrics = []

    train_stat_metrics_1 = server.test_model(model_no=1, clients_to_test = clients, set_to_use='train')
    print("Model 1 train")
    metrics_return = print_metrics(train_stat_metrics_1, num_samples_1, prefix='train_')
    all_metrics = all_metrics + metrics_return
    
    print("Model 2 train")
    train_stat_metrics_2 = server.test_model(model_no=2, clients_to_test = clients, set_to_use='train')
    metrics_return = print_metrics(train_stat_metrics_2, num_samples_2, prefix='train_')
    all_metrics = all_metrics + metrics_return

    #writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'

    print("Model 1 test")
    test_stat_metrics_1 = server.test_model(model_no=1, clients_to_test = clients, set_to_use=eval_set)
    metrics_return = print_metrics(test_stat_metrics_1, num_samples_1, prefix='{}_'.format(eval_set))
    all_metrics = all_metrics + metrics_return

    print("Model 2 test")
    test_stat_metrics_2 = server.test_model(model_no=2, clients_to_test = clients, set_to_use=eval_set)
    metrics_return = print_metrics(test_stat_metrics_2, num_samples_2, prefix='{}_'.format(eval_set))
    all_metrics = all_metrics + metrics_return

    return all_metrics

    #writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None

    # average metrics to be returned
    metrics_return = []

    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]

        metrics_return.append(np.average(ordered_metric, weights=ordered_weights))

        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))

    return metrics_return

if __name__ == '__main__':
    main()
