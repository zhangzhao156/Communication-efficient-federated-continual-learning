# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 19:29
# @Author  : zhao
# @File    : main_mnist_fedavg.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from collections import Iterable  # < py38
import copy
from net_fewc import CNNMnist,CNNFashion_Mnist
import logging
import gzip
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert a list of list to a list [[],[],[]]->[,,]
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name) # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

def load_data(data_folder, data_name, label_name):
    """
        data_folder: 文件目录
        data_name： 数据文件名
        label_name：标签数据文件名
    """
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # Generates random samples from all_idexs,return a array with size of num_items
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid6(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 60, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels#.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 6, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid5(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 50, 1200
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels#.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 5, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid4(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 40, 1500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels#.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid3(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 30, 2000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels#.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid2(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 20, 3000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels#.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid1(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels#.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def test_img(net_g, datatest):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_pred = []
    data_label = []
    data_loader = DataLoader(datatest, batch_size=test_BatchSize, shuffle=True)
    l = len(data_loader)
    loss = torch.nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data).to(device), Variable(target).type(torch.LongTensor).to(device)
        # data, target = Variable(data), Variable(target).type(torch.LongTensor)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += loss(log_probs, target).item()
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.detach().max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.detach().view_as(y_pred)).long().cpu().sum()
        data_pred.append(y_pred.cpu().detach().data.tolist())
        data_label.append(target.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('all_precision',all_precision,'all_recall',all_recall,'all_fscore',all_fscore)
    # print(classification_report(list_data_label, list_data_pred))
    print(confusion_matrix(list_data_label, list_data_pred))
    # print('test_loss', test_loss)
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} {:.2f}'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    # logging.info('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} {:.2f}\n'.format(
    #     test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def getGradShapes(Model):
    """Return the shapes and sizes of the weight matrices"""
    gradShapes = []
    gradSizes = []
    for n, p in Model.named_parameters():
        gradShapes.append(p.data.shape)
        gradSizes.append(np.prod(p.data.shape))
    return gradShapes, gradSizes


def getGradVec(w):
    """Return the gradient flattened to a vector"""
    gradVec = []
    # flatten
    # for n, p in Model.named_parameters():
    #     # gradVec.append(torch.zeros_like(p.data.view(-1)))
    #     gradVec.append(p.grad.data.view(-1).float())
    for k in w.keys():
        # gradVec.append(torch.zeros_like(p.data.view(-1)))
        gradVec.append(w[k].view(-1).float())
    # concat into a single vector
    gradVec = torch.cat(gradVec)
    return gradVec


def setGradVec(Model, vec):
    """Set the gradient to vec"""
    # put vec into p.grad.data
    vec = vec.to(device)
    gradShapes, gradSizes = getGradShapes(Model=Model)
    startPos = 0
    i = 0
    for n, p in Model.named_parameters():
        shape = gradShapes[i]
        size = gradSizes[i]
        i += 1
        # assert (size == np.prod(p.grad.data.size()))
        p.grad.data.zero_()
        p.grad.data.add_(vec[startPos:startPos + size].reshape(shape))
        startPos += size


def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)
    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec ** 2)[1][-k:]
    # _, topkIndices = torch.topk(vec**2, k)
    ret[topkIndices] = vec[topkIndices]
    return ret, topkIndices


def quantize(x):
    compress_settings = {'n': 32}
    # compress_settings.update(input_compress_settings)
    # assume that x is a torch tensor

    n = compress_settings['n']
    # print('n:{}'.format(n))
    x = x.float()
    x_norm = torch.norm(x, p=float('inf'))  # inf_norm = max(abs(x))

    sgn_x = ((x > 0).float() - 0.5) * 2

    p = torch.div(torch.abs(x), x_norm)
    renormalize_p = torch.mul(p, n)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)
    final_p = renormalize_p - floor_p
    margin = (compare < final_p).float()
    xi = (floor_p + margin) / n

    Tilde_x = x_norm * sgn_x * xi

    return Tilde_x


def quantize_log(x):
    compress_settings = {'n': 16}
    # compress_settings.update(input_compress_settings)
    # assume that x is a torch tensor
    n = compress_settings['n']
    # print('n:{}'.format(n))
    x = x.float()
    x_norm = torch.norm(x, p=float('inf'))  # inf_norm = max(abs(x))
    sgn_x = ((x > 0).float() - 0.5) * 2
    p = torch.div(torch.abs(x), x_norm)
    lookup = torch.linspace(0, -10, n)
    log_p = torch.log2(p)
    round_index = [(torch.abs(lookup - k)).min(dim=0)[1] for k in log_p]
    round_p = [2 ** (lookup[i]) for i in round_index]
    round_p = torch.stack(round_p).to(device)
    # print('round_p',round_p)
    # print('x_norm',x_norm)

    Tilde_x = x_norm * round_p * sgn_x

    return Tilde_x


def quantization_layer(sizes, x):
    q_x = torch.zeros_like(x)
    startPos = 0
    for i in sizes:
        q_x[startPos:startPos + i] = quantize(x[startPos:startPos + i])
        # q_x[startPos:startPos + i] = quantize_log(x[startPos:startPos + i])
        startPos += i
    return q_x


def sparsity(fisher, w_update, w_prev, topkIndices):
    Shapes = []
    Sizes = []
    for j in fisher.keys():
        Shapes.append(fisher[j].shape)
        Sizes.append(np.prod(fisher[j].shape))
    # print('fisher sizes', Sizes)
    fisher_vector = getGradVec(fisher)
    fisher_vector_spar = torch.zeros_like(fisher_vector)
    fisher_vector_spar[topkIndices] = fisher_vector[topkIndices]
    # fisher_vector_spar_q = quantization_layer(sizes=torch.tensor([144, 16, 4608, 32, 2560, 5]),
    #                                           x=fisher_vector_spar)
    fisher_vector_spar_q = quantize(fisher_vector_spar)
    model_vector_spar_q = w_update + w_prev
    # model_vector_spar_q = w_update - w_prev
    fisher_spar = {k: torch.zeros_like(fisher[k]) for k in fisher.keys()}
    model_spar = {k: torch.zeros_like(fisher[k]) for k in fisher.keys()}
    startPos = 0
    j = 0
    for k in fisher.keys():
        shape = Shapes[j]
        size = Sizes[j]
        j += 1
        fisher_spar[k] = fisher_vector_spar_q[startPos:startPos + size].reshape(shape).double()
        model_spar[k] = model_vector_spar_q[startPos:startPos + size].reshape(shape).double()
        startPos += size
    return fisher_spar, model_spar

def consolidate(Model, Weight, MEAN_pre, epsilon):
    OMEGA_current = {n: p.data.clone().zero_() for n, p in Model.named_parameters()}
    for n, p in Model.named_parameters():
        p_current = p.detach().clone()
        p_change = p_current - MEAN_pre[n]
        # W[n].add_((p.grad**2) * torch.abs(p_change))
        # OMEGA_add = W[n]/ (p_change ** 2 + epsilon)
        # W[n].add_(-p.grad * p_change)
        OMEGA_add = torch.max(Weight[n], Weight[n].clone().zero_()) / (p_change ** 2 + epsilon)
        # OMEGA_add = Weight[n] / (p_change ** 2 + epsilon)
        # OMEGA_current[n] = OMEGA_pre[n] + OMEGA_add
        OMEGA_current[n] = OMEGA_add
    MEAN_current = {n: p.data for n, p in Model.named_parameters()}
    return OMEGA_current, MEAN_current


# FL + EWC
if __name__ == '__main__':
    # logging.basicConfig(filename='./20200512_cicids_our_noniid1_E_1_T_1.log', level=logging.DEBUG)
    # logging.info('11111')
    epsilon = 0.0001
    rho = 1.0 #0.5
    Lamda = 1.0 #0.5
    E = 5
    T = 50 #50
    ## FedAvg
    # rho = 1.0
    # Lamda = 0.0
    frac = 1.0
    num_clients = 10
    batch_size = 512
    test_BatchSize = 32

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ### MNIST
    # dataset_train = DealDataset('./data/MNIST/raw', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    #                             transform=trans_mnist)
    # dataset_test = DealDataset('./data/MNIST/raw', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
    #                            transform=trans_mnist)
    dataset_train = DealDataset('./fashion', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                                transform=trans_mnist)
    dataset_test = DealDataset('./fashion', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                               transform=trans_mnist)

    # save_global_model = 'save_model.pkl'

    # # IID Data
    # dict_clients = iid(dataset_train, num_users=num_clients)
    # dict_clients = mnist_noniid3(dataset_train, num_users=num_clients)
    dict_clients = mnist_noniid2(dataset_train, num_users=num_clients)
    # dict_clients = mnist_noniid1(dataset_train, num_users=num_clients)

    # net_global = CNNMnist(lamda=Lamda).to(device) #.double()
    net_global = CNNFashion_Mnist(lamda=Lamda).to(device)  # .double()
    
    # for n, p in net_global.named_parameters():
    #   p.data.zero_()
    w_glob = net_global.state_dict()
    # print(w_glob)
    crit = torch.nn.CrossEntropyLoss()#torch.DoubleTensor weight=torch.FloatTensor([1, 1.2, 1.2, 1.2, 3]).to(device)
    # optimizer = torch.optim.SGD(net_global.parameters(), lr=0.001, momentum=0.5)
    net_global.train()

    for interation in range(T):
        w_locals, loss_locals = [], []
        # print('interationh',interation)
        for client in range(num_clients):
            # net = CNN(N_class=3,lamda=10000).double().to(device)
            net = copy.deepcopy(net_global).to(device)
            # crit = torch.nn.CrossEntropyLoss()
            net.train()
            # opt_net = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
            # opt_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
            # opt_net = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) ### IID
            opt_net = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.7) ##, momentum=0.5
            # opt_net = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.7)  ##, momentum=0.5

            print('interation', interation, 'client', client)
            idx_traindataset = DatasetSplit(dataset_train, dict_clients[client])
            ldr_train = DataLoader(idx_traindataset, batch_size=512, shuffle=True)
            dataset_size = len(ldr_train.dataset)
            epochs_per_task = E
            mean_pre = {n: p.clone().detach() for n, p in net.named_parameters()}
            t0 = time.clock()
            for epoch in range(1, epochs_per_task + 1):
                correct = 0
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    images, labels = Variable(images).to(device), Variable(labels).type(torch.LongTensor).to(device)
                    net.zero_grad()
                    scores = net(images)
                    ce_loss = crit(scores, labels)
                    loss = ce_loss
                    pred = scores.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
                    loss.backward()
                    opt_net.step()

                Accuracy = 100. * correct.type(torch.FloatTensor) / dataset_size
                # print('Train Epoch:{}\tLoss:{:.4f}\tProx_Loss:{:.4f}\tCE_Loss:{:.4f}\tAccuracy: {:.4f}'.format(epoch,loss.item(),prox_loss.item(),ce_loss.item(),Accuracy))
                print('Train Epoch:{}\tLoss:{:.4f}\tCE_Loss:{:.4f}\tAccuracy: {:.4f}'.format(epoch,loss.item(),ce_loss.item(),Accuracy))
                # print(classification_report(labels.cpu().data.view_as(pred.cpu()), pred.cpu()))

            w_locals.append(copy.deepcopy(net.state_dict()))
            t1 = time.clock()
            print('client:\t', client, 'trainingtime:\t', str(t1 - t0))
        w_glob = FedAvg(w_locals)
        net_global.load_state_dict(w_glob)
        net_global.eval()
        acc_test, loss_test = test_img(net_global, dataset_test)
        print("Testing accuracy: {:.2f}".format(acc_test))

    model_dict = net_global.state_dict()  # 自己的模型参数变量
    test_dict = {k: w_glob[k] for k in w_glob.keys() if k in model_dict}  # 去除一些不需要的参数
    model_dict.update(test_dict)  # 参数更新
    net_global.load_state_dict(model_dict)  # 加载

    # for n, p in net_global.named_parameters():
    #     p = w_glob[n]

    # net_global.load_state_dict(w_glob)
    net_global.eval()
    acc_test, loss_test = test_img(net_global, dataset_test)
    print("Testing accuracy: {:.2f}".format(acc_test))