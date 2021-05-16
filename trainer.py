import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from args import read_options
from tensorboardX import SummaryWriter
from scipy.sparse import csc_matrix
import torch.nn as nn
from utils import *
from net import *
import time
import os
from networkx.algorithms.link_analysis import pagerank
import operator
import math
from neighbor_matcher import Neighbor_Matcher


class Trainer(object):
    
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v) 
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True 

        self.load_embed()

        self.neighbor_matcher = Neighbor_Matcher(self.edge_matrix, self.edge_nums, self.rel_emb, self.ent_emb, self.embed_dim, self.kernel_num, self.device).to(self.device)
        self.T_GRUA =T_GRUA(self.kernel_num, self.embed_dim, self.hidden_dim, self.h_hrt_bg,self.ent2id, self.id2ent, self.id2rel, self.batch_size,self.edge_matrix,self.edge_nums,self.topk, self.rel_emb, self.ent_emb, self.device).to(self.device)

        model_params = list(set(list(self.T_GRUA.parameters()) + list(self.neighbor_matcher.parameters())))

        self.parameters = filter(lambda p: p.requires_grad, model_params)
        '''
        model_dict = self.Glstmnn.state_dict()
        for k, v in model_dict.items(): 
            print("model_dict:",k) 
        '''
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.writer = SummaryWriter('logs/' + self.prefix) #initial


    def loadtxt(self,path):
        name2ids = {}
        with open(path) as file:
            for i, line in enumerate(file):
                line = line.strip().split()
                name,idx = line
                name2ids[name] = int(idx)
        return name2ids
    
    def load_embedding(self,path):
        def load_from_file(path,i):
            embeds = []
            with open(path) as file:
                for line in file:
                    line = line.strip().split()
                    embeds.append(list(map(float, line)))
                    i = i+1
            return embeds,i
        i = 0
        relation_embeds,i = load_from_file(path,i)
        relation_embeds = torch.tensor(relation_embeds)
        return relation_embeds

    def load_embed(self):
        rel_bg = json.load(open(self.dataset + '/relation2ids')) 
        ent_all = json.load(open(self.dataset+'/ent2ids'))
        train_tasks = json.load(open(self.dataset+'/train_tasks.json'))
        test_tasks = json.load(open(self.dataset+'/test_tasks.json'))
        dev_tasks = json.load(open(self.dataset+'/dev_tasks.json'))
        ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
        rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)
        if self.embed_model=='ComplEx':
            ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
            ent_std = np.std(ent_embed, axis=1, keepdims=True)
            rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
            rel_std = np.std(rel_embed, axis=1, keepdims=True)
            eps = 1e-3
            ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
            rel_embed = (rel_embed - rel_mean) / (rel_std + eps)
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json')) 
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json')) 
        train_rel = train_tasks.keys()
        test_rel = test_tasks.keys()
        dev_rel = list(dev_tasks.keys())
        bg_rel = rel_bg.keys()
 

        rel2id = {}
        ent2id = {}
        rel_embedding = []
        ent_embedding = []
        i = 0
        for key in rel_bg.keys():
            rel2id[key] = i
            i=i+1
            rel_embedding.append(list(rel_embed[rel_bg[key],:]))
        
        for rel in list(train_rel)+list(test_rel)+list(dev_rel):
            rel2id[rel] = i
            i=i+1
        j = 0
        for key in ent_all.keys():
            ent2id[key] = j
            j = j + 1
            ent_embedding.append(list(ent_embed[ent_all[key],:]))

        rel_embedding = torch.tensor(rel_embedding)
        ent_embedding = torch.tensor(ent_embedding)

        self.bg_rel_id_list = []
        self.train_rel_id_list = []
        self.test_rel_id_list = []
        self.dev_rel_id_list = []
        for i in range(len(list(train_rel))):
            self.train_rel_id_list.append(rel2id[list(train_rel)[i]])
        for i1 in range(len(list(test_rel))):
            self.test_rel_id_list.append(rel2id[list(test_rel)[i1]])
        for i2 in range(len(list(dev_rel))):
            self.dev_rel_id_list.append(rel2id[list(dev_rel)[i2]])
        for i3 in range(len(list(bg_rel))):
            self.bg_rel_id_list.append(rel2id[list(bg_rel)[i3]])

        facts_data = []
        pg_facts_data = []
        bg_data = []
        with open(self.dataset+'/path_graph') as file:
            for line in file:
                fact = line.strip().split()
                #print("fact:",fact)
                pg_facts_data.append([ent2id[fact[0]],rel2id[fact[1]],ent2id[fact[2]]])
                pg_facts_data.append([ent2id[fact[2]],rel2id[fact[1]+'_inv'],ent2id[fact[0]]])
                bg_data.append([ent2id[fact[0]],ent2id[fact[2]],rel2id[fact[1]]])
                bg_data.append([ent2id[fact[2]],ent2id[fact[0]],rel2id[fact[1]+'_inv']])
        file.close()
        id2rel = {v: k for k, v in rel2id.items()}
        id2ent = {v: k for k, v in ent2id.items()}

        with open(os.path.join(self.dataset, 'pagerank.txt')) as file:
            self.pagerank = list(map(lambda x: float(x.strip()), file.readlines()))
        # edge_matrix
        self.edge_data = [[] for _ in range(len(ent2id) + 1)]
        for fact in bg_data:
            e1,e2,rel = fact
            self.edge_data[e1].append((e1, e2, rel))
        for head in range(len(self.edge_data)):
            self.edge_data[head].sort(key=lambda x: self.pagerank[x[1]], reverse=True) # pagerank
            self.edge_data[head] = self.edge_data[head][:self.neighbor_limit]
        self.edge_nums = torch.tensor(list(map(len, self.edge_data)), dtype=torch.long)
        edge_entities = [list(map(lambda x: x[1], edges)) for edges in self.edge_data]
        edge_relations = [list(map(lambda x: x[2], edges)) for edges in self.edge_data]
        edge_entities = list2tensor(edge_entities, padding_idx=len(ent2id), dtype=torch.int, device=self.device)
        edge_relations = list2tensor(edge_relations, padding_idx=len(rel2id), dtype=torch.int,device=self.device)
        self.edge_matrix = torch.stack((edge_entities, edge_relations), dim=2)

        train_trip_id = reltri2tri(train_tasks, rel2id, ent2id)
        test_trip_id = reltri2tri(test_tasks,rel2id,ent2id)
        dev_trip_id = reltri2tri(dev_tasks,rel2id,ent2id)

        self.rel_emb = nn.Embedding(len(rel2id.keys())+1, self.embed_dim).to(self.device) 
        self.rel_emb.weight.data[:len(rel_bg)] = rel_embedding 
        self.rel_emb.weight.data[-1] = torch.zeros(1,self.embed_dim)
        self.ent_emb = nn.Embedding(len(ent2id.keys())+1,self.embed_dim).to(self.device)
        self.ent_emb.weight.data[:len(ent2id)] = ent_embedding
        self.ent_emb.weight.data[-1] = torch.zeros(1,self.embed_dim)

        self.rel2id = rel2id
        self.ent2id = ent2id
        self.id2rel = id2rel
        self.id2ent = id2ent
        self.train_tasks = train_tasks
        self.bg_data = bg_data

        self.train_trip_id = train_trip_id
        self.test_trip_id = test_trip_id 
        self.dev_trip_id = dev_trip_id
        self.pg_facts_data = pg_facts_data 
        self.h_hrt_bg = h2hrt(self.pg_facts_data)

        self.rel_test_trip = trip2rel2tripid(self.test_trip_id,self.test_rel_id_list)
        self.rel_dev_trip = trip2rel2tripid(self.dev_trip_id,self.dev_rel_id_list)
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json')) 
        self.rel_bg_trip = trip2rel2tripid(self.pg_facts_data, self.bg_rel_id_list)

        rel_emb_n = self.rel_emb.weight.data[:-1]
        rel_emb_bro = rel_emb_n.unsqueeze(0).repeat(rel_emb_n.size()[0],1,1)
        cos_rel_all = torch.sigmoid(torch.cosine_similarity(rel_emb_bro,rel_emb_n.unsqueeze(1),dim=-1))
        pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.cos_rel_all = pad(cos_rel_all)

        path_dict_str = json.load(open(self.dataset+'/train_valid_test_pair2paths_name.json')) # train_test_pair2paths_name.json
        self.trian_test_path, self.train_test_path_id = path_read(path_dict_str, self.rel2id, self.ent2id)
        self.test2relkind_dict, self.test2relkind = test_relkind(self.rel_test_trip, self.train_test_path_id)
        self.set_rel_sim = set_rel_sim_count(self.test2relkind)

    def save(self, path=None):
        if not path:
            path = self.save_path
        state_all = {'neighbor_matcher':self.neighbor_matcher.state_dict(),'T_GRUA':self.T_GRUA.state_dict()}
        torch.save(state_all, path)

    def load(self):
        checkpoint = torch.load(self.save_path+'_hits10_best')
        self.neighbor_matcher.load_state_dict(checkpoint['neighbor_matcher'])
        self.T_GRUA.load_state_dict(checkpoint['T_GRUA'])


    def train(self):
        logging.info('START TRAINING...')
        self.T_GRUA.train()
        self.neighbor_matcher.train()

        batch_num = 0
        best_hits10 = 0.0

        for data in train_generate(self.few, self.neg_num, self.dataset, self.batch_size, self.train_tasks,self.ent2id,self.rel2id, self.id2ent, self.id2rel, self.e1rel_e2, self.rel2candidates): #产生每一轮训练用的数据

            batch_num = batch_num+1
            self.optim.zero_grad()

            #path
            support_pair, query_pair, one_tomany_train, candidates_id = data[0]
            #t-h
            support_left, support_right, query_left, query_right, false_left, false_right = data[1]
            support_pair_name = []
            for i in range(len(support_pair)):
                support_pair_name.append([self.id2ent[support_pair[i][0]], self.id2ent[support_pair[i][1]]])
            query_pair_name = []
            for i in range(len(query_pair)):
                query_pair_name.append([self.id2ent[query_pair[i][0]], self.id2ent[query_pair[i][1]]])

            support_rel = rel_submit(support_pair,self.train_test_path_id) 
            support_path = path_submit(support_pair, self.train_test_path_id) 
            query_path = path_submit(query_pair, self.train_test_path_id)
            
            query_head = [pair[0] for pair in query_pair]
            query_tail = [pair[1] for pair in query_pair]
            query_head = torch.tensor(query_head)
            query_tail = torch.tensor(query_tail)

            loss_path = self.T_GRUA(support_pair, support_rel, support_path, query_head, query_tail, one_tomany_train, self.cos_rel_all, True, candidates_id)

            query_scores = self.neighbor_matcher(support_left, query_left, support_right, query_right) # batch_size
            false_scores = self.neighbor_matcher(support_left, false_left, support_right, false_right) # |batch_size*neg_sample_num|, eg: 16*3=48
            query_scores = query_scores.view(self.batch_size,-1).repeat(1,self.neg_num).view(-1) # |batch_size*neg_sample_num|, eg: 16*3=48
            margin_ = query_scores - false_scores
            loss_th = F.relu(self.margin - margin_).mean()

            if loss_path==0:
                loss = loss_th
            else:
                loss = loss_path + loss_th #joint-train

            loss.backward()  
            self.optim.step()
            print("loss:",loss)

            with torch.no_grad():
                #If you think it will take a long time to test, you can set this value higher, but you may miss some better values
                if batch_num % self.eval_every==0 :  #NELL-ONE:50  WIKI-ONE:200 or 500
                    hit1, hit5, hit10, mrr = self.eval(self.mode)
                    self.writer.add_scalar('hit1', hit1, batch_num)
                    self.writer.add_scalar('hit5', hit5, batch_num)
                    self.writer.add_scalar('hit10', hit10, batch_num)
                    self.writer.add_scalar('mrr', mrr, batch_num)
                    self.T_GRUA.train()

                    self.save()  #store
                    if hit10 > best_hits10:
                        self.save(self.save_path + '_hits10_best')
                        best_hits10 = hit10
            if batch_num==self.max_batches:  
                self.save()
                break



    def test_(self, mode='test'):
        logging.info('Pre-trained model loaded')
        self.load()
        self.eval(mode)

    # according to the score # ns-shot
    def find_sqs(self,key): 
        pair_key = self.test2relkind_dict[key]
        score = self.set_rel_sim[key]
        index,score = zip(*sorted(enumerate(score), key=operator.itemgetter(1),reverse=True))

        support_pair = []
        query_paireval = []
        for i in range(len(pair_key)):
            if i < self.few:
                support_pair.append([pair_key[index[i]][0],pair_key[index[i]][1]])
            else:
                query_paireval.append([pair_key[index[i]][0],pair_key[index[i]][1]])

        return support_pair, query_paireval

    # baseline # n-shot
    def find_sq(self,key,mode): 
        if mode=='test':
            trip_key = self.rel_test_trip[key]
        else:
            trip_key = self.rel_dev_trip[key]
        
        support_pair = []
        query_paireval = []
        for i in range(len(trip_key)):
            if i < self.few:
                support_pair.append([trip_key[i][0],trip_key[i][2]])
            else:
                query_paireval.append([trip_key[i][0],trip_key[i][2]])
        return support_pair, query_paireval


    def eval(self, mode='test'):
        logging.info('EVALUATING ON %s DATA' % mode.upper())
        self.T_GRUA.eval()
        self.neighbor_matcher.eval()

        hit1_sum = []
        hit5_sum = []
        hit10_sum = []
        mrr_sum = []

        if mode == 'test':
            rel_trip = self.rel_test_trip
        else:
            rel_trip = self.rel_dev_trip

        for key, values in rel_trip.items():

            logging.info('key:{}'.format(key)) 
            logging.info('values_len:{}'.format(len(values)))
            if len(values)<2:
                break
            str_rel = self.id2rel[key]
            candidate_ent = self.rel2candidates[str_rel]
            candidate_ent_id = []
            for i in range(len(candidate_ent)):
                candidate_ent_id.append(self.ent2id[candidate_ent[i]])

            support_pair_eval, query_paireval = self.find_sq(key, mode) 
            support_name = []
            for i in range(len(support_pair_eval)):
                support_name.append([self.id2ent[support_pair_eval[i][0]],self.id2ent[support_pair_eval[i][1]]])
            support_rel = rel_submit(support_pair_eval,self.train_test_path_id) 
            support_path = path_submit(support_pair_eval, self.train_test_path_id) 

            hit1, hit5, hit10, mrr = self.eval_score(key, candidate_ent, candidate_ent_id, support_pair_eval, query_paireval, support_rel, support_path)
            hit1_sum = hit1_sum + hit1
            hit5_sum = hit5_sum + hit5
            hit10_sum = hit10_sum + hit10
            mrr_sum = mrr_sum + mrr
            logging.critical('one------Hits1:{:.3f}, Hits5:{:.3f}, Hits10:{:.3f}, MRR:{:.3f}'.format(np.mean(hit1), np.mean(hit5), np.mean(hit10), np.mean(mrr)))

        logging.critical('All------Hits1:{:.3f}, Hits5:{:.3f}, Hits10:{:.3f}, MRR:{:.3f}'.format(np.mean(hit1_sum), np.mean(hit5_sum), np.mean(hit10_sum), np.mean(mrr_sum)))
        self.T_GRUA.train()
        self.neighbor_matcher.train()
        return np.mean(hit1_sum), np.mean(hit5_sum), np.mean(hit10_sum), np.mean(mrr_sum)

    def divide_query(self, query_triples):
        if len(query_triples) <= self.batch_size:
            return [query_triples]
        else:
            subs = []
            num = int(len(query_triples)/self.batch_size)
            idx = 0
            for i in range(num):
                subs.append(query_triples[idx:idx+self.batch_size])
                idx += self.batch_size
            
            if idx < len(query_triples):
                subs.append(query_triples[idx:])
            return subs

    def eval_score(self, key, candidates, candidate_ent_id, support_pair, eval_pair, support_rel, support_path):
        head = []
        right_tail = []
        support_pair_name = []
        for i in range(len(support_pair)):
            support_pair_name.append([self.id2ent[support_pair[i][0]], self.id2ent[support_pair[i][1]]])
        support_left = [pair[0] for pair in support_pair]
        support_right = [pair[1] for pair in support_pair]

        i= 0
        for i in range(len(eval_pair)):
            head.append(eval_pair[i][0])
            right_tail.append(eval_pair[i][1])
        one2many_list_all = []
        for i in range(len(head)):
            one2many = self.e1rel_e2[self.id2ent[int(head[i])]+self.id2rel[int(key)]] 
            one2many2id = [self.ent2id[_] for _ in one2many]
            one2many2id.remove(right_tail[i])
            one2many_list_all.append(one2many2id)
        head = torch.tensor(head)
        right_tail = torch.tensor(right_tail)
        num = head.size()[0]
        num_count = math.ceil(num/float(self.batch_size))
        hits1 = []
        hits5 = []
        hits10 = []
        mrrs = []

        for i in range(num_count):
            if i == num_count-1:
                head_batch = head[i*self.batch_size:]
                right_tail_batch = right_tail[i*self.batch_size:]
                one2many = one2many_list_all[i*self.batch_size:]
                score_batch_list, node_batch_list = self.T_GRUA(support_pair, support_rel, support_path, head_batch, right_tail_batch, one2many, self.cos_rel_all, False, candidate_ent_id) #16个

                for m in range(head_batch.size()[0]):
                    query_right = [int(right_tail_batch[m])]
                    for ent in candidates:
                        if (ent not in self.e1rel_e2[self.id2ent[int(head_batch[m])]+self.id2rel[key]]) and ent != self.id2ent[int(right_tail_batch[m])]:
                            query_right.append(self.ent2id[ent])
                    scores = []
                    sub_query_right = self.divide_query(query_right)
                    for sub in sub_query_right:
                        sub_query_left = [int(head_batch[m])] * len(sub)
                        score = self.neighbor_matcher(support_left, sub_query_left, support_right, sub).tolist()
                        scores += score
                    #'''
                    #add-path-score
                    if len(score_batch_list)!=0:
                        graph_score_one = score_batch_list[m]
                        graph_node_one = node_batch_list[m]
                        for n in range(len(graph_node_one)):
                            loc = query_right.index(graph_node_one[n])
                            scores[loc] = max(scores[loc], graph_score_one[n])
                    #'''
                    scores = np.array(scores)
                    sort = list(np.argsort(-scores))
                    top_scores = []
                    for idx in sort[:123]:
                        top_scores.append(scores[idx])

                    rank = sort.index(0) + 1
                    if rank <= 10:
                        hits10.append(1.0)
                    else:
                        hits10.append(0.0)
                    if rank <= 5:
                        hits5.append(1.0)
                    else:
                        hits5.append(0.0)
                    if rank <= 1:
                        hits1.append(1.0)
                    else:
                        hits1.append(0.0)
                    mrrs.append(1.0/rank)

            else:
                head_batch = head[i*self.batch_size : (i+1)*self.batch_size]
                right_tail_batch = right_tail[i*self.batch_size : (i+1)*self.batch_size]
                one2many = one2many_list_all[i*self.batch_size : (i+1)*self.batch_size]

                score_batch_list, node_batch_list = self.T_GRUA(support_pair, support_rel, support_path, head_batch, right_tail_batch, one2many, self.cos_rel_all, False, candidate_ent_id)

                for m in range(head_batch.size()[0]):
                    query_right = [int(right_tail_batch[m])]
                    for ent in candidates:
                        if (ent not in self.e1rel_e2[self.id2ent[int(head_batch[m])]+self.id2rel[key]]) and ent != self.id2ent[int(right_tail_batch[m])]:
                            query_right.append(self.ent2id[ent])

                    scores = []
                    sub_query_right = self.divide_query(query_right)
                    for sub in sub_query_right:
                        sub_query_left = [int(head_batch[m])] * len(sub)

                        score = self.neighbor_matcher(support_left, sub_query_left, support_right, sub).tolist()
                        scores += score
                    #'''
                    #add-path-score
                    if len(score_batch_list)!=0:
                        graph_score_one = score_batch_list[m]
                        graph_node_one = node_batch_list[m]
                        for n in range(len(graph_node_one)):
                            loc = query_right.index(graph_node_one[n])
                            scores[loc] = max(scores[loc], graph_score_one[n])  
                    #'''                            
                    scores = np.array(scores)
                    sort = list(np.argsort(-scores))
                    rank = sort.index(0) + 1

                    if rank <= 10:
                        hits10.append(1.0)
                    else:
                        hits10.append(0.0)
                    if rank <= 5:
                        hits5.append(1.0)
                    else:
                        hits5.append(0.0)
                    if rank <= 1:
                        hits1.append(1.0)
                    else:
                        hits1.append(0.0)
                    mrrs.append(1.0/rank)

        return hits1,hits5,hits10,mrrs






