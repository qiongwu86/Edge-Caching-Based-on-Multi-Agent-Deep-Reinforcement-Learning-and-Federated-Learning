import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--slots', type=int, default=1000,
                        help="number of slots of training")
    parser.add_argument('--clients_num', type=int, default=10,
                        help="number of clients: K")
    parser.add_argument('--cachesize', type=list, default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                        help="size of cache: CS")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=256,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='AAE', help='model name')

    # workspace arguments
    parser.add_argument('--clean_dataset', type=bool, default=True, help="clean\
                        the model/data_set or not")
    parser.add_argument('--clean_user', type=bool, default=True, help="clean\
                        the user/ or not")
    parser.add_argument('--clean_clients', type=bool, default=True, help="clean\
                        the model/clients or not")

    # data set
    parser.add_argument('--dataset', type=str, default='ml-1m', help="name of dataset")

    parser.add_argument('--decay', type=float, default=0.95,
                        help='Asy updata decay (default: 0.95)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Asy updata gamma (default: 0.5)')
    parser.add_argument('--update_decay', type=float, default=0.99,
                        help='Asy update_decay (default: 0.99)')

    # other arguments
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args
