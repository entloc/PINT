import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import json
import random
from utils import *


class Neighbor_Matcher(nn.Module):

    def __init__(self, edge_data, edge_nums, rel_emb, ent_emb, emb_dim, kernel_num, device):
        """
        edge_data  : outgoing edges in background graph
        edge_nums  : outgoing degree in background graph
        rel_emb    : embeddings of relations
        ent_emb    : embeddings of entities
        emb_dim    : dimension of the embeddings
        kernel_num : number of kernels
        mu,sigma   : params of normal distribution
        """
        nn.Module.__init__(self)

        self.edge_data = edge_data
        self.edge_nums = edge_nums
        self.rel_emb = rel_emb
        self.ent_emb = ent_emb
        self.embed_dim = emb_dim
        self.padding_idx = len(rel_emb.weight.data) - 1
        self.device = device

        tensor_mu = self.kernel_mus(kernel_num)
        tensor_sigma = self.kernel_sigmas(kernel_num)
        if self.device:
            tensor_mu = tensor_mu.to(self.device)
            tensor_sigma = tensor_sigma.to(self.device)
        self.mu = Variable(tensor_mu, requires_grad = False).view(1,1,1,kernel_num)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1,1,1,kernel_num)
        
        project_num = 200
        self.dense_3 = nn.Linear(kernel_num, 1, 1)
        self.dense_2 = nn.Linear(project_num, 1, 1)
        self.dense_1 = nn.Linear(kernel_num, project_num, 1)

        self.attention_w = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        #self.attention_b = nn.Parameter(torch.FloatTensor(1)).to(self.device)

    def kernel_mus(self,n_kernels):
        l_mu = [1] # for exact match.
        if n_kernels == 1:
            return l_mu

        bin_size = 1.0 / (n_kernels - 1)  # score range from [0, 1]
        l_mu.append(1 - bin_size / 2)     # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)

        return torch.FloatTensor(l_mu)

    def kernel_sigmas(self,n_kernels):
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.1] * (n_kernels - 1)

        return torch.FloatTensor(l_sigma)

    def get_neighbor_mask(self, support, query):
        support_neighbors = self.edge_data[support].select(-1,1) # support_num * neighbor_limit

        query_neighbors = self.edge_data[query].select(-1,1)     # batch_size * neighbor_limit

        support_size = support_neighbors.size()
        query_size = query_neighbors.size()

        support_mask = torch.where(support_neighbors==self.padding_idx,torch.zeros(support_size).to(self.device),torch.ones(support_size).to(self.device)) 
        # same as support_neighbors, eg: 5 * 500 
        query_mask = torch.where(query_neighbors==self.padding_idx,torch.zeros(query_size).to(self.device),torch.ones(query_size).to(self.device)) 
        # same as query_neighbors, eg: 16 * 500
        
        return support_neighbors,query_neighbors,support_mask,query_mask

    def get_attention_weights(self, heads, tails, tail_neighbors):
        head_embed = self.ent_emb(torch.tensor(heads).to(self.device))            # few/batch_size * emb_dim 
        tail_embed = self.ent_emb(torch.tensor(tails).to(self.device))            # few/batch_size * emb_dim 
        type_embed = tail_embed - head_embed                                      # few/batch_size * emb_dim 
        tail_neighbor_embed = self.ent_emb(tail_neighbors.long().to(self.device)) # few/batch_size * neighbor_limit * emb_dim
        
        type_embed = self.attention_w(type_embed)
        
        type_embed = type_embed.view(type_embed.size()[0],1,type_embed.size()[1]) # 5 * 1 * 100

        similarity = torch.bmm(type_embed,torch.transpose(tail_neighbor_embed,1,2)).squeeze(1) # 5 * 500
        
        
        return similarity

    def get_intersect_matrix(self, s_embed, q_embed, mask_s, mask_q, s_attention, q_attention,q_neighbor_num):
        
        similarity = torch.bmm(q_embed, torch.transpose(s_embed, 1, 2)) # batch_size * query_neighbor_num * support_neighbor_num 
        similarity = similarity * q_attention * s_attention
        similarity = similarity.view(q_embed.size()[0], q_embed.size()[1], s_embed.size()[1], 1)  
        
        # max
        pooling_max, _ = similarity.topk(k=1,dim=2)   # batch_size * query_neighbor_num * 1 * 1
        pooling_value = torch.exp((- ((pooling_max - self.mu) ** 2) / (self.sigma ** 2) / 2))  # batch_size * query_neighbor_num * 1 * kernel_num
        pooling_value = torch.squeeze(pooling_value,2)  # batch_size * query_neighbor_num * kernel_num
        
        # sum
        log_pooling_value = torch.log(torch.clamp(pooling_value, min=1e-30)) * mask_q * 0.01  # batch_size * query_neighbor_num * kernel_num
        log_pooling_sum = torch.sum(log_pooling_value, dim=1)
        return log_pooling_sum
        

    def similarity_encoder(self, support_neighbors, query_neighbors, mask_s, mask_q, s_attention, q_attention,q_neighbor_num):
        s_embed = self.rel_emb(support_neighbors.long().to(self.device)) # batch_size * neighbor_num * embed_dim
        q_embed = self.rel_emb(query_neighbors.long().to(self.device))   # batch_size * neighbor_num * embed_dim
        s_embed_norm = F.normalize(s_embed, 2, 2)
        q_embed_norm = F.normalize(q_embed, 2, 2)

        mask_s = mask_s.view(mask_s.size()[0], 1, mask_s.size()[1], 1) # batch_size * 1 * neighbor_num * 1
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)    # batch_size * neighbor_num * 1

        s_attention = s_attention.view(s_attention.size()[0],1,s_attention.size()[1]) 
        q_attention = q_attention.view(q_attention.size()[0],q_attention.size()[1],1) 

        log_pooling_sum = self.get_intersect_matrix(s_embed_norm, q_embed_norm, mask_s, mask_q, s_attention, q_attention,q_neighbor_num) # batch_size * kernel_num

        return log_pooling_sum

    def forward(self, support_heads, query_heads, support_tails, query_tails):

        support_tail_neighbors, query_tail_neighbors, mask_s_t, mask_q_t = self.get_neighbor_mask(support_tails,query_tails)
        
        query_neighbor_num = self.edge_nums[query_tails] 

        support_attention = self.get_attention_weights(support_heads,support_tails,support_tail_neighbors) 
        query_attention = self.get_attention_weights(query_heads,query_tails,query_tail_neighbors) 

        # calculate support pairs respectively 
        for i in range(support_tail_neighbors.size(0)):
            cur_support_tail_neighbors = support_tail_neighbors[i].repeat(query_tail_neighbors.size(0),1) 
            cur_mask_s_t = mask_s_t[i].repeat(query_tail_neighbors.size(0),1)
            cur_support_attention = support_attention[i].repeat(query_tail_neighbors.size(0),1) 

            tail_enc = self.similarity_encoder(cur_support_tail_neighbors, query_tail_neighbors, cur_mask_s_t, mask_q_t, cur_support_attention, query_attention, query_neighbor_num)

            score = torch.squeeze(torch.sigmoid(self.dense_2(self.dense_1(tail_enc))),1) # batch_size
            
            if i==0:
                score_all = score.unsqueeze(0)
                kernel_all = tail_enc.unsqueeze(0)
            else:
                score_all = torch.cat([score_all,score.unsqueeze(0)],dim = 0) # few * batch_size
                kernel_all = torch.cat([kernel_all,tail_enc.unsqueeze(0)],dim = 0) # few * batch_size * kernel_num
        
        # max
        output = torch.max(score_all,0)[0] # batch_size
        
        return output # 1D: |batch_size|
        