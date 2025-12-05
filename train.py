import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from constraints import ConstraintManager, StartNodeConstraint
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
import create_graphs


def train_vae_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to('cuda' if args.cuda else 'cpu')
        y = Variable(y).to('cuda' if args.cuda else 'cpu')

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred,z_mu,z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss_bce.item(), loss_kl.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_'+args.fname, loss_bce.item(), epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' +args.fname, loss_kl.item(), epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)

def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time = 1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)


    return G_pred_list


def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to('cuda' if args.cuda else 'cpu'), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list



def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to('cuda' if args.cuda else 'cpu')
        y = Variable(y).to('cuda' if args.cuda else 'cpu')

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.item(), epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)


def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)


    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list



def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to('cuda' if args.cuda else 'cpu'), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:,i:i+1,:].to('cuda' if args.cuda else 'cpu'), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to('cuda' if args.cuda else 'cpu')
        y = Variable(y).to('cuda' if args.cuda else 'cpu')

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j+1,y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:,j,0:end_idx], y[:,j,0:end_idx])*end_idx


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.item(), epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)





## too complicated, deprecated
# def test_mlp_partial_bfs_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
#         for i in range(max_num_node):
#             # 1 back up hidden state
#             hidden_prev = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to('cuda' if args.cuda else 'cpu'), current=i, y_len=y_len, sample_time=sample_time)
#             y_pred_long[:, i:i + 1, :] = x_step
#
#             rnn.hidden = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
#
#             print('finish node', i)
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()
#
#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output, label_embedding=None, label_head=None, time_head=None, **kwargs):

    rnn.train()
    output.train()
    loss_sum = 0
    # Initialize Constraint Manager
    constraint_manager = None
    if 'id_to_label' in kwargs:
        id_to_label = kwargs['id_to_label']
        label_to_id = {v: k for k, v in id_to_label.items()}
        constraint_manager = ConstraintManager(args.config, label_to_id)

    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        
        # Handle labels
        if label_embedding is not None:
            x_label_unsorted = data['x_label'].long().squeeze(2) # (batch, len)
            y_label_unsorted = data['y_label'].long().squeeze(2) # (batch, len)
            x_label_unsorted = x_label_unsorted[:, 0:y_len_max]
            y_label_unsorted = y_label_unsorted[:, 0:y_len_max]
            
        # Handle times
        if time_head is not None:
            x_time_unsorted = data['x_time'].float() # (batch, len, 3)
            y_time_unsorted = data['y_time'].float() # (batch, len, 3)
            x_time_unsorted = x_time_unsorted[:, 0:y_len_max, :]
            y_time_unsorted = y_time_unsorted[:, 0:y_len_max, :]

            
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        
        if label_embedding is not None:
            x_label = torch.index_select(x_label_unsorted, 0, sort_index)
            y_label = torch.index_select(y_label_unsorted, 0, sort_index)
            
        if time_head is not None:
            x_time = torch.index_select(x_time_unsorted, 0, sort_index)
            y_time = torch.index_select(y_time_unsorted, 0, sort_index)
            if args.cuda:
                x_time = x_time.cuda()
                y_time = y_time.cuda()


        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x)
        y = Variable(y)
        output_x = Variable(output_x)
        output_y = Variable(output_y)
        
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            output_x = output_x.cuda()
            output_y = output_y.cuda()
            if label_embedding is not None:
                x_label = x_label.cuda()
                y_label = y_label.cuda()

        # Embed labels and concatenate
        if label_embedding is not None:
            x_label_embed = label_embedding(x_label) # (batch, len, embed_size)
            x = torch.cat((x, x_label_embed), dim=2) # (batch, len, prev_node + embed_size)

        h = rnn(x, pack=True, input_len=y_len)
        
        # Label Loss
        label_loss = 0
        if label_head is not None:
            # h is (batch, len, hidden)
            # Pack h to match y_label packing
            packed_h = pack_padded_sequence(h, y_len, batch_first=True)
            label_logits = label_head(packed_h.data)
            y_label_packed = pack_padded_sequence(y_label, y_len, batch_first=True).data
            label_loss = F.cross_entropy(label_logits, y_label_packed)
            
            # Constraint Loss
            if constraint_manager:
                start_idx = 0
                # Iterate through packed sequence steps. 
                # batch_sizes gives the number of active sequences at each time step.
                for step, step_batch_size in enumerate(packed_h.batch_sizes):
                    end_idx = start_idx + step_batch_size.item()
                    step_logits = label_logits[start_idx:end_idx]
                    
                    # Determine which sequences have this as their final step
                    # y_len is a list of sequence lengths (sorted descending)
                    # If y_len[i] == step+1, then this is the final step for sequence i
                    is_final_for_batch = [(step + 1) == length for length in y_len[:step_batch_size.item()]]
                    
                    # Compute constraint loss for the current step
                    # Pass is_final_step flag for sequences that end at this step
                    if any(is_final_for_batch):
                        # Create a tensor mask for final steps
                        for i, is_final in enumerate(is_final_for_batch):
                            if is_final and i < step_logits.size(0):
                                # Apply FinalNodeConstraint to this specific sequence
                                c_loss = constraint_manager.compute_loss(
                                    step_logits[i:i+1], None, step, is_final_step=True
                                )
                                label_loss += c_loss
                    
                    # Apply regular step constraints (StartNode, EndNode, ParallelNode)
                    # We need to pass full adjacency and time info for ParallelNodeConstraint
                    # y is (batch, len, max_prev_node) - this is the ground truth adjacency
                    # y_time is (batch, len, 3) - ground truth times
                    # time_pred is (batch, 3) for the current step (if we computed it per step)
                    
                    # Wait, time_pred is computed for the WHOLE sequence at once in line 559?
                    # No, line 559 is OUTSIDE the loop.
                    # I need to compute time_pred INSIDE the loop or pass the slice.
                    
                    # Actually, let's compute time_pred for the current step inside the loop
                    # h is packed, so we can access h for the current step
                    
                    current_time_pred = None
                    if time_head is not None:
                        # step_h = h[start_idx:end_idx] # This is hard because h is packed
                        # But packed_h.data is (total_steps, hidden)
                        # step_logits corresponds to packed_h.data[start_idx:end_idx]
                        # So we can compute time_pred for this step
                        step_h = packed_h.data[start_idx:end_idx]
                        current_time_pred = time_head(step_h)
                    
                    c_loss = constraint_manager.compute_loss(
                        step_logits, None, step, is_final_step=False,
                        time_pred=current_time_pred,
                        adj=y, # Full adjacency
                        time_gt=y_time # Full ground truth times
                    )
                    label_loss += c_loss

                    
                    start_idx = end_idx

        # Time Loss
        time_loss = 0
        if time_head is not None:
            # h is (batch, len, hidden)
            # Pack h to match y_time packing
            packed_h = pack_padded_sequence(h, y_len, batch_first=True)
            time_pred = time_head(packed_h.data)
            y_time_packed = pack_padded_sequence(y_time, y_len, batch_first=True).data
            time_loss = F.mse_loss(time_pred, y_time_packed)


        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))
        if args.cuda:
            idx = idx.cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1)))
        if args.cuda:
            hidden_null = hidden_null.cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        
        if label_head is not None:
            loss += args.label_loss_weight * label_loss
            
        if time_head is not None:
            # Default weight 1.0 if not specified
            time_weight = getattr(args, 'time_loss_weight', 1.0)
            loss += time_weight * time_loss

            
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if batch_idx % args.epochs_log == 0:
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.item(), epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)



def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16, label_embedding=None, label_head=None, time_head=None, id_to_label=None):

    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to('cuda' if args.cuda else 'cpu') # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
    
    # Label initialization
    if label_embedding is not None:
        # Force SOS to be 'START' if available
        if id_to_label:
            start_id = None
            for k, v in id_to_label.items():
                if v == 'START':
                    start_id = k
                    break
            sos_idx = start_id if start_id is not None else args.num_node_labels - 1
        else:
            sos_idx = args.num_node_labels - 1
        
        x_label_step = Variable(torch.ones(test_batch_size, 1) * sos_idx).long().to('cuda' if args.cuda else 'cpu')
        
        # Store predicted labels
        pred_labels = torch.zeros(test_batch_size, max_num_node).long()
        
        # Store predicted times
        pred_times = torch.zeros(test_batch_size, max_num_node, 3).float()

        
        # Track sequence lengths (for early termination on END)
        lengths = torch.ones(test_batch_size, dtype=torch.long) * max_num_node
        finished = torch.zeros(test_batch_size, dtype=torch.bool)
        
        # Find END label ID
        end_label_id = None
        if id_to_label:
            for k, v in id_to_label.items():
                if v == 'END':
                    end_label_id = k
                    break
        
        # Initialize Constraint Manager
        constraint_manager = None
        if id_to_label:
            label_to_id = {v: k for k, v in id_to_label.items()}
            constraint_manager = ConstraintManager(args.config, label_to_id)
        
    for i in range(max_num_node):
        
        if label_embedding is not None:
            x_label_embed = label_embedding(x_label_step) # (batch, 1, embed)
            x_step_input = torch.cat((x_step, x_label_embed), dim=2)
        else:
            x_step_input = x_step
            
        h = rnn(x_step_input)
        
        # Predict label for this node (i)
        if label_head is not None:
            label_logits = label_head(h) # (batch, 1, num_classes)
            
            # Sample label
            label_probs = F.softmax(label_logits, dim=2)
            # dist = torch.distributions.Categorical(label_probs.squeeze(1))
            # sampled_label = dist.sample() # (batch)
            # Or just argmax for testing? Or sampling?
            # Sampling is better for diversity.
            sampled_label = torch.multinomial(label_probs.view(-1, args.num_node_labels), 1).view(test_batch_size, 1)
            
            pred_labels[:, i] = sampled_label.squeeze(1)
            
            # Check for END token (early termination)
            if end_label_id is not None:
                for batch_idx in range(test_batch_size):
                    if not finished[batch_idx]:
                        label_id = sampled_label[batch_idx].item()
                        if label_id == end_label_id:
                            finished[batch_idx] = True
                            lengths[batch_idx] = i + 1
            
            # Prepare for next step
            x_label_step = sampled_label
            
        # Predict time for this node (i)
        if time_head is not None:
            time_pred = time_head(h) # (batch, 1, 3)
            pred_times[:, i, :] = time_pred.squeeze(1)

        
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).to('cuda' if args.cuda else 'cpu')
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).to('cuda' if args.cuda else 'cpu')
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).to('cuda' if args.cuda else 'cpu')
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).to('cuda' if args.cuda else 'cpu')
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to('cuda' if args.cuda else 'cpu')
    y_pred_long_data = y_pred_long.data.long()

    # Generate graphs
    G_pred_list = []
    for i in range(test_batch_size):
        # Use dynamic length if labels were predicted, otherwise use full length
        if label_head is not None:
            length = lengths[i].item()
        else:
            length = max_num_node
        
        # Slice predictions based on actual length
        adj_pred = decode_adj(y_pred_long_data[i, :length, :].cpu().numpy())
        G_pred = get_graph(adj_pred)
        # Attach labels
        if label_head is not None:
            labels = pred_labels[i, :length].cpu().numpy()
            
            # Re-implement get_graph logic here to keep indices consistent
            # Use the sliced adjacency matrix that matches the actual length
            adj_full = decode_adj(y_pred_long_data[i, :length, :].cpu().numpy())
            # Identify valid nodes (non-zero rows)
            valid_idx = np.where(~np.all(adj_full == 0, axis=1))[0]
            
            # Assign labels
            # Assign labels and times
            for node_idx, idx in enumerate(valid_idx):

                # node_idx is index in adj_full (0 to length-1)
                # idx is the new node index in G_pred (0 to num_nodes-1)
                
                # Assign Label
                if (idx) < len(labels):
                    G_pred.nodes[node_idx]['label'] = int(labels[idx])
                else:
                    G_pred.nodes[node_idx]['label'] = 0
                    
                # Assign Times
                if time_head is not None:
                    times = pred_times[i, :length].detach().cpu().numpy()
                    if idx < len(times):
                        G_pred.nodes[node_idx]['norm_time'] = float(times[idx][0])
                        G_pred.nodes[node_idx]['trace_time'] = float(times[idx][1])
                        G_pred.nodes[node_idx]['prev_event_time'] = float(times[idx][2])
        
        G_pred_list.append(G_pred)

    return G_pred_list




def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to('cuda' if args.cuda else 'cpu')
        y = Variable(y).to('cuda' if args.cuda else 'cpu')
        output_x = Variable(output_x).to('cuda' if args.cuda else 'cpu')
        output_y = Variable(output_y).to('cuda' if args.cuda else 'cpu')
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to('cuda' if args.cuda else 'cpu')
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).to('cuda' if args.cuda else 'cpu')
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.item(), epoch*args.batch_ratio+batch_idx)
        # print(y_pred.size())
        feature_dim = y_pred.size(0)*y_pred.size(1)
        loss_sum += loss.item()*feature_dim/y.size(0)
    return loss_sum/(batch_idx+1)


########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output, label_embedding=None, label_head=None, time_head=None, id_to_label=None):

    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    epoch = 1
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)
    
    # Add label parameters to optimizer if present
    if label_embedding is not None:
        optimizer_rnn.add_param_group({'params': label_embedding.parameters()})
    if label_head is not None:
        optimizer_output.add_param_group({'params': label_head.parameters()})
    if time_head is not None:
        optimizer_output.add_param_group({'params': time_head.parameters()})


    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output, label_embedding=label_embedding, label_head=label_head, time_head=time_head, id_to_label=id_to_label)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size, label_embedding=label_embedding, label_head=label_head, time_head=time_head, id_to_label=id_to_label)

                    
                    # Filter graphs
                    G_pred_step = [g for g in G_pred_step if args.min_gen_node_count <= g.number_of_nodes() <= args.max_gen_node_count]
                    
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                
                # save graphs as txt
                fname_txt = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.txt'
                save_graph_list_txt(G_pred, fname_txt, id_to_label)
                
                # draw graphs
                fname_fig = args.figure_prediction_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time)
                draw_graph_list(G_pred[:16], 4, 4, fname=fname_fig)
                
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)


########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))

    for sample_time in range(1,4):
        if 'GraphRNN_MLP' in args.note:
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        if 'GraphRNN_VAE' in args.note:
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')


########### for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
        f.write('train,test\n')
        for iter in range(max_iter):
            if 'GraphRNN_MLP' in args.note:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            if 'GraphRNN_RNN' in args.note:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            print('train',nll_train,'test',nll_test)
            f.write(str(nll_train)+','+str(nll_test)+'\n')

    print('NLL evaluation done')
