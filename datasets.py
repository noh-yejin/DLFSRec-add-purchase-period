import random
import torch
import tqdm
import numpy as np
from torch.utils.data import Dataset


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def get_pad(args,user_seq, max_len):
    result = []

    # if args.period_add_like_cxt==True: 
        # for seq in tqdm.tqdm(user_seq):
        #     ids, cxt,period = [], [],[]
        #     for item_id, act_time in seq[:-1]:
        #         ids.append(item_id)
        #         cxt.append(act_time)
        #     # period.append(act_time[-1])  # add item period
        #     # pad_num = max_len - len(ids)
        #     # if pad_num == max_len: # all padding
        #     #     result.append([np.zeros(max_len, dtype=int), np.zeros((max_len, 6)), np.zeros((max_len,1))])
        #     # elif pad_num <= 0:
        #     #     result.append([np.array(ids[-max_len:]), np.array(cxt[-max_len:]),np.array(period[-max_len:])])
        #     # else:
        #     #     ids = [0] * pad_num + ids
        #     #     period=[0]*pad_num + period
        #     #     result.append([np.array(ids), np.pad(np.array(cxt), ((pad_num, 0), (0, 0))),np.array(period)])
        #     pad_num = max_len - len(ids)
        #     if pad_num == max_len:  # all padding
        #         result.append([np.zeros(max_len, dtype=int), np.zeros((max_len, 7))])
        #     elif pad_num <= 0:  # cut
        #         result.append([np.array(ids[-max_len:]), np.array(cxt[-max_len:])])
        #     else:  # rest padding
        #         ids = [0] * pad_num + ids
        #         result.append([np.array(ids), np.pad(np.array(cxt), ((pad_num, 0), (0, 0)))])

    for seq in tqdm.tqdm(user_seq):

        ids, cxt = [], []  

        if args.period_add_like_cxt==True: 
            for item_id, act_time in seq[:-1]:
                # import IPython; IPython.embed(colors="Linux"); exit(1)
                ids.append(item_id)
                cxt.append(act_time)  # if period_add_like_cxt=true, cxt and period
            pad_num = max_len - len(ids)
            if pad_num == max_len:  # all padding
                result.append([np.zeros(max_len, dtype=int), np.zeros((max_len, 7))])
            elif pad_num <= 0:  # cut
                result.append([np.array(ids[-max_len:]), np.array(cxt[-max_len:])])
            else:  # rest padding
                ids = [0] * pad_num + ids
                result.append([np.array(ids), np.pad(np.array(cxt), ((pad_num, 0), (0, 0)))])
        else: 
            for item_id, act_time in seq[:-1]:    
                ids.append(item_id)
                cxt.append(act_time)  # if period_add_like_cxt=true, cxt and period
            pad_num = max_len - len(ids)
            if pad_num == max_len:  # all padding
                result.append([np.zeros(max_len, dtype=int), np.zeros((max_len, 6))])
            elif pad_num <= 0:  # cut
                result.append([np.array(ids[-max_len:]), np.array(cxt[-max_len:])])
            else:  # rest padding
                ids = [0] * pad_num + ids
                result.append([np.array(ids), np.pad(np.array(cxt), ((pad_num, 0), (0, 0)))])



    return result


class DLFSRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.data_type = data_type
        self.max_len = args.max_seq_length

        if data_type == 'train':
            for seq in user_seq:  
                input_ids = seq[-(self.max_len + 2):-2] # leave 2 out
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])  
        elif data_type == 'valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1]) # leave 1 out
        else:
            self.user_seq = user_seq # all sequence
        


        self.test_neg_items = test_neg_items
        self.data_type = data_type

        self.inputs = get_pad(args,self.user_seq, self.max_len)


    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index): # error

        if self.args.period_add_like_cxt==True:  #period add

            items = self.user_seq[index]

            input_ids, input_cxt_period= self.inputs[index]
            
            assert len(input_ids) == self.max_len
            assert len(input_cxt_period) == self.max_len

            pos_id, pos_cxt_period = items[-1] # last

            seq_set = set([x[0] for x in items])
            neg_id = neg_sample(seq_set, self.args.item_size) # random item
            neg_cxt_period = pos_cxt_period  # only id contrastive? pos_id<->neg_id

            inputs = {
                'uid': torch.tensor(index, dtype=torch.long),
                'id': torch.tensor(input_ids, dtype=torch.long),
                'cxt': torch.tensor(input_cxt_period, dtype=torch.float32),
            }
            pos = {
                'pos_id': torch.tensor(pos_id, dtype=torch.long),
                'pos_cxt': torch.tensor(pos_cxt_period, dtype=torch.float32),
            }
            neg = {
                'neg_id': torch.tensor(neg_id, dtype=torch.long),
                'neg_cxt': torch.tensor(neg_cxt_period, dtype=torch.float32),
            }

            if self.test_neg_items is not None:
                
                test_samples_id = self.test_neg_items[index]  # error
                test_samples_cxt = [pos_cxt_period for x in test_samples_id]

                test_samples = {
                    'test_samp_id': torch.tensor(test_samples_id, dtype=torch.long),
                    'test_samp_cxt': torch.tensor(test_samples_cxt, dtype=torch.float32),
                }
                cur_tensors = (inputs, pos, neg, test_samples)
            else:
                cur_tensors = (inputs, pos, neg)

        else:  # baseline

            items = self.user_seq[index]
            input_ids, input_cxt = self.inputs[index]

            assert len(input_ids) == self.max_len
            assert len(input_cxt) == self.max_len


            pos_id, pos_cxt = items[-1]
            
            seq_set = set([x[0] for x in items])
            neg_id = neg_sample(seq_set, self.args.item_size)  # select non-interaction item
            neg_cxt = pos_cxt  # cross entropy->only use id
            inputs = {
                'uid': torch.tensor(index, dtype=torch.long),
                'id': torch.tensor(input_ids, dtype=torch.long),
                'cxt': torch.tensor(input_cxt, dtype=torch.float32)
            }
            pos = {
                'pos_id': torch.tensor(pos_id, dtype=torch.long),
                'pos_cxt': torch.tensor(pos_cxt, dtype=torch.float32)
            }
            neg = {
                'neg_id': torch.tensor(neg_id, dtype=torch.long),
                'neg_cxt': torch.tensor(neg_cxt, dtype=torch.float32)
            }

            if self.test_neg_items is not None:
                test_samples_id = self.test_neg_items[index]  # error
                test_samples_cxt = [pos_cxt for x in test_samples_id]


                test_samples = {
                    'test_samp_id': torch.tensor(test_samples_id, dtype=torch.long),
                    'test_samp_cxt': torch.tensor(test_samples_cxt, dtype=torch.float32)
                    
                }

                cur_tensors = (inputs, pos, neg, test_samples)
            else:
                cur_tensors = (inputs, pos, neg)
        return cur_tensors
