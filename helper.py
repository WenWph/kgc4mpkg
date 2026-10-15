import numpy as np
import pickle
import torch
import random
import dgl

def load_data(prefix='dataset/'):
    type_mask = np.load(prefix + 'node_types.npy')
    train_val_test_pos = np.load(prefix + 'train_val_test_pos.npz')
    # in_file = open(prefix + 'dic_dic_triple.pickle', 'rb')
    # dic_dic_triple = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + 'dic_no_count.pickle', 'rb')
    dic_no_count = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'dic_no_offset.pickle', 'rb')
    dic_no_offset = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + 'dic_no_path.pickle', 'rb')
    # dic_no_path = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'dic_no_dic_path_instance.pickle', 'rb')
    dic_no_dic_path_instance = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'dic_no_label_rel.pickle', 'rb')
    dic_no_label_rel = pickle.load(in_file)
    in_file.close()

    return type_mask, train_val_test_pos, dic_no_count, dic_no_offset, dic_no_dic_path_instance, dic_no_label_rel


class batch_generator:
    def __init__(self, batch_size, num_link=None, indices=None, shuffle=True):
        if num_link is not None:
            self.num_data = num_link
            self.indices = np.arange(num_link)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_batch_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_batch(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_batch_left(self):
        return self.num_batch() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


def id2no(offset_list, x):
    assert offset_list, "offset_list不能为空"
    for off_i, offset in enumerate(offset_list):
        if x < offset:
            return off_i - 1
    return len(offset_list)-1


def parse_batch(dic_path_instance, batch_triple, device, if_mask = True):
    # list_list_g, list_list_instance, list_list_idx_mapped = [[], []], [[], []], [[], []]
    # a 'g' represents a path
    list_g, list_path_no, list_instance, list_idx_mapped = [], [], [], []

    for path_no, array_instance in dic_path_instance.items():
        set_node = set()
        list_num_neigh = []
        list_instance_sample = []
        list_idx = []
        list_edge = []
        for ent in range(2):  # head or tail
            for row in batch_triple:
                if row[-ent] not in set_node:
                    set_node.add(row[-ent])
                    array_instance_select = array_instance[array_instance[:, 0] == row[-ent]]
                    if if_mask:
                        mask = [False if array_single.tolist() in batch_triple else True for array_single in array_instance_select[:, 0:3]]
                        list_instance_sample.append(array_instance_select[mask])
                        list_num_neigh.append(len(array_instance_select[mask]))
                        list_edge += [(row[-ent], i) for i in range(len(array_instance_select[mask]))]
                    else:
                        list_instance_sample.append(array_instance_select)
                        list_num_neigh.append(len(array_instance_select))
                        list_edge += [(row[-ent], i) for i in range(len(array_instance_select))]
        mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(set_node))}
        # let the sequence of edges' source nodes aligned with nodes in set_node
        sorted_index = sorted(range(len(list_edge)), key=lambda i: list_edge[i])
        list_instance_sample = torch.LongTensor(np.vstack(list_instance_sample)[sorted_index]).to(device)

        num_node_in_batch = len(set_node)
        # indexes of object nodes (representing instances), following nodes in set_node
        list_source = [i + num_node_in_batch for i in range(sum(list_num_neigh))]
        g = dgl.graph(([], [])).to(device)
        g.add_nodes(num_node_in_batch + sum(list_num_neigh))
        g.add_edges(list_source, [mapping[tup[0]] for tup in list_edge])
        list_g.append(g)
        list_instance.append(list_instance_sample)
        for ent in range(2):  # head or tail
            for row in batch_triple:
                list_idx.append(mapping[row[-ent]])
        list_idx_mapped.append(np.array(list_idx))
        list_path_no.append(path_no)


    return list_g, list_instance, list_idx_mapped


def generate_pos_neg_triples(pos_triple, num_entity, triple_list):
    neg_triple = []
    for [head, relation, tail] in pos_triple:
        prob = np.random.random()
        while True:
            if prob <= 0.5:
                head = random.choice(np.arange(num_entity))
            else:
                tail = random.choice(np.arange(num_entity))
            # if (head, tail, relation) not in self.train_triples：
            if [head, relation, tail] not in triple_list:
                break
        neg_triple.append([head, relation, tail])
    return neg_triple


class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss_last = np.Inf
        self.path = path

    def __call__(self, val_loss, model):

        if val_loss < self.val_loss_min:
            # print(f'\t\t\t\t\t\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}')
            print(f'\t\t\t\t\t\t\t\t\t\t\t\t\t\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}')
            self.val_loss_min = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)

        elif val_loss < self.val_loss_last:
            self.counter = 0

        else: # 新的验证损失未减小
            self.counter += 1
            # print(f'\t\t\t\t\t\tEarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'\t\t\t\t\t\t\t\t\t\t\t\t\t\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        self.val_loss_last = val_loss
