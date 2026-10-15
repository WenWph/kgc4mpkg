import time
from helper import *
from model1 import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_path = 'best_model.pt'
type_mask, train_val_test_pos, dic_no_count, dic_no_offset, dic_no_dic_path_instance, dic_no_label_rel = load_data()

dic_path_instance = {}
# form {(path): instance_array}
for no in dic_no_dic_path_instance.keys():
    dic_path_instance.update(dic_no_dic_path_instance[no])

pos_train = train_val_test_pos['train_pos']
pos_valid = train_val_test_pos['val_pos']
pos_test = train_val_test_pos['test_pos']
list_g, list_instance, list_idx_mapped = parse_batch(dic_path_instance, pos_train.tolist(), device, False)

num_entity = 0
for k, count in dic_no_count.items():
    num_entity += count
num_type = len(dic_no_count)
num_relation = len(dic_no_label_rel)
list_count = [dic_no_count[0], dic_no_count[1]]
list_offset = [dic_no_offset[0], dic_no_offset[1]]
list_all_triple = np.vstack((pos_train, pos_valid, pos_test)).tolist()

list_feature = []
dim_in = []
# 为每个实体构造one-hot特征向量，维度为所属类别中实体数量
for i in range(num_type):
    dim_type = np.sum(type_mask == i)
    dim_in.append(dim_type)
    # 矩阵坐标，均为(i,i)对角线元素
    inx_temp = torch.LongTensor(np.vstack((np.arange(dim_type), np.arange(dim_type))))
    value_temp = torch.FloatTensor(np.ones(dim_type))
    # 生成稀疏矩阵，对角线元素均为1
    list_feature.append(torch.sparse_coo_tensor(inx_temp, value_temp, torch.Size([dim_type, dim_type])).to(device))

dropout = 0.3
learning_rate = 0.001
# candidate_lr = [0.0005, 0.001, 0.005, 0.01]
weight_decay = 1e-5
dim_embed = 200
dim_hidden = 128
num_head = 5
num_layer = 1
# num_layer = [1,2,3]
num_epoch = 100
comp_op = 'mul'
# candidate_op = ['sub', 'mul', 'corr', 'mlp']
lambda_self = 0.5
batch_size = 16
# candidate_b = [8, 16, 32, 64]
decoder = 'conve'
# candidate_d = ['conve', 'distmult', 'transe']

for repeat in range(1):

    repeat_no = repeat + 1
    print(f'第{repeat_no}次')

    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MPKGE_LP(dim_embed, dim_hidden, num_relation, dim_in, num_layer,24,  num_head, lambda_self, dropout, comp_op, decoder)
    model.to(device)
    idx_generator_train = batch_generator(batch_size=batch_size, num_link=len(pos_train))
    idx_generator_val = batch_generator(batch_size=batch_size, num_link=len(pos_valid), shuffle=False)
    params = [value.numel() for value in model.parameters()]
    print(f'参数种类{len(params)}')
    # print(f'参数数量{params}')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=5, path=save_path)

    print(f'\n\nbatch: {batch_size}')
    print(f'num_layer: {num_layer}')
    print(f'learning_rate: {learning_rate}')
    print(f'head: {num_head}')
    print(f'lambda: {lambda_self}')
    print(f'operation: {comp_op}')
    print(f'decoder: {decoder}')

    time_start = time.time()
    for epoch in range(num_epoch):
        time_start_epoch = time.time()
        model.train()
        loss_epoch_train = 0
        # tri = pos_train[0]
        # for i, tri in enumerate(pos_train):
        for batch_no in range(idx_generator_train.num_batch()):  # 迭代次数，即批次数，数据量÷单个批次内数量
            # time_start_batch = time.time()
            optimizer.zero_grad()
            idx_batch_pos_train = idx_generator_train.next()  # 返回一个批次的index
            idx_batch_pos_train.sort()
            batch_pos_train = pos_train[idx_batch_pos_train].tolist()
            batch_neg_train = generate_pos_neg_triples(batch_pos_train, num_entity, list_all_triple)
            batch_pos_neg_train = np.array(batch_pos_train + batch_neg_train)
            label_train = torch.cat([torch.ones(len(batch_pos_train)), torch.zeros(len(batch_neg_train))]).to(device)
            emb_ent, emb_rel = model(list_g, list_instance, list_idx_mapped, type_mask, list_feature, pos_train, device)
            # time_end_batch = time.time()
            score_train = model.cal_score(emb_ent, emb_rel, batch_pos_neg_train)
            # score_pos, score_neg = torch.chunk(score_train, chunks=2)
            # loss_batch = -torch.mean(nn.functional.logsigmoid(score_pos) + nn.functional.logsigmoid(score_neg))
            loss_batch = model.bce_loss(torch.sigmoid(score_train), label_train)
            loss_epoch_train += loss_batch
            loss_batch.backward()
            optimizer.step()
        time_end_epoch = time.time()
        loss_epoch_train = loss_epoch_train / idx_generator_train.num_batch()

        model.eval()
        with torch.no_grad():
            # time_val_start = time.time()
            loss_epoch_val = 0
            for batch_no in range(idx_generator_val.num_batch()):  # 迭代次数，即批次数，数据量÷单个批次内数量
                # time_batch_start = time.time()
                idx_batch_pos_val = idx_generator_val.next()  # 返回一个批次的index
                idx_batch_pos_val.sort()
                batch_pos_val = pos_train[idx_batch_pos_val].tolist()
                batch_neg_val = generate_pos_neg_triples(batch_pos_val, num_entity, list_all_triple)
                batch_pos_neg_val = np.array(batch_pos_val + batch_neg_val)
                label_val = torch.cat([torch.ones(len(batch_pos_val)), torch.zeros(len(batch_neg_val))]).to(device)
                embed_ent_val, embed_rel_val = model(list_g, list_instance, list_idx_mapped, type_mask, list_feature, pos_train, device)
                score_val = model.cal_score(embed_ent_val, embed_rel_val, batch_pos_neg_val)
                loss_batch = model.bce_loss(torch.sigmoid(score_val), label_val)
                loss_epoch_val += loss_batch
            # time_val_end = time.time()
            loss_epoch_val = loss_epoch_val / idx_generator_val.num_batch()
            # print(f'Ep {epoch + 1:03d} Train {loss_epoch_train:.10f} | Dur {time_end_epoch - time_start_epoch:.2f} | Valid {loss_epoch_val} | Dur {time_val_end - time_val_start:.2f}')
            print(f'Ep {epoch + 1:03d} Train {loss_epoch_train:.10f} | Dur {time_end_epoch - time_start_epoch:.2f} | Valid {loss_epoch_val}')

            early_stopping(loss_epoch_val, model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

    time_end = time.time()
    print(f'Time{repeat_no:02d} Train+Valid for {epoch + 1:03d} epoches | Dur {(time_end - time_start)/60:.2f} mins')
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print('----- Start testing -----')
    with torch.no_grad():
        list_k = [1, 3, 10]
        mean_ranks_tail, mrrs_tail, mean_ranks_head, mrrs_head = [], [], [], []
        hits_tail, hits_head = [], []
        list_hit_at_k_tail, list_hit_at_k_head = [], []
        for _ in list_k:
            hits_tail.append([])
            hits_head.append([])

        test_id = 0
        for fact in pos_test:
            test_id += 1
            # time_test_triple_begin = time.time()
            [h_id, r_id, t_id] = fact
            test_input_tail = [[h_id, r_id, x] for x in np.arange(num_entity)]
            test_input_head = [[x, r_id, t_id] for x in np.arange(num_entity)]
            test_input = np.array(test_input_tail + test_input_head)
            embed_ent_test, embed_rel_test = model(list_g, list_instance, list_idx_mapped, type_mask, list_feature, pos_train, device)
            score_test = model.cal_score(embed_ent_test, embed_rel_test, test_input)
            score_test_tail, score_test_head = torch.chunk(score_test, chunks=2)

            rank_tail = int((score_test_tail.argsort(descending=True) == t_id).nonzero().view(-1))
            rank_head = int((score_test_head.argsort(descending=True) == h_id).nonzero().view(-1))

            # time_test_triple_end = time.time()
            # print(f'Test_triple {test_id}/{len(pos_test)} | Dur {time_test_triple_end - time_test_triple_begin:.2f}')
            print(f'Test_triple {test_id}/{len(pos_test)} | rank_tail: {rank_tail + 1}')
            print(f'Test_triple {test_id}/{len(pos_test)} | rank_head: {rank_head + 1}')
            mean_ranks_tail.append(rank_tail + 1)
            mean_ranks_head.append(rank_head + 1)
            mrrs_tail.append(1. / (rank_tail + 1))
            mrrs_head.append(1. / (rank_head + 1))
            for k_i, k_value in enumerate(list_k):
                hits_tail[k_i].append(rank_tail + 1 <= k_value)
                hits_head[k_i].append(rank_head + 1 <= k_value)

        mean_rank_tail = float(torch.tensor(mean_ranks_tail, dtype=torch.float).mean())
        mean_rank_head = float(torch.tensor(mean_ranks_head, dtype=torch.float).mean())
        mrr_tail = float(torch.tensor(mrrs_tail, dtype=torch.float).mean())
        mrr_head = float(torch.tensor(mrrs_head, dtype=torch.float).mean())
        for hit_tail, hit_head in zip(hits_tail, hits_head):
            list_hit_at_k_tail.append(int(torch.tensor(hit_tail).sum()) / len(hit_tail))
            list_hit_at_k_head.append(int(torch.tensor(hit_head).sum()) / len(hit_head))

        print(f'\n')
        print(f'Mean Rank')
        print(f'Mean reciprocal rank:')
        for k_value in list_k:
            print(f'Hit@{k_value}')
        print(f'\n')
        print('Tail prediction')
        print(mean_rank_tail)
        print(mrr_tail)
        for hit_at_k in list_hit_at_k_tail:
            print(hit_at_k)
        print('Head prediction')
        print(mean_rank_head)
        print(mrr_head)
        for hit_at_k in list_hit_at_k_head:
            print(hit_at_k)
        print(f'\n')
        print(f'Time{repeat_no:02d} Train+Valid Dur {(time_end - time_start)/60:.2f} mins')
        time_end_end = time.time()
        print(f'Time{repeat_no:02d} Test Dur {(time_end_end - time_end) / 60:.2f} mins\n\n\n\n')

        file_path = f'output/batch{batch_size}_layer{num_layer}_head{num_head}_lambda{lambda_self}_{comp_op}_{decoder}_第{repeat_no:02d}次.txt'
        file = open(file_path, 'w', encoding='utf-8')
        try:
            file.write(f"Mean Rank\n")
            file.write(f"Mean reciprocal rank\n")
            for k_value in list_k:
                file.write(f"Hit@{k_value}\n")
            file.write(f'Tail prediction\n')
            file.write(f'{mean_rank_tail}\n')
            file.write(f'{mrr_tail}\n')
            for hit_at_k in list_hit_at_k_tail:
                file.write(f'{hit_at_k}\n')
            file.write(f'Head prediction\n')
            file.write(f'{mean_rank_head}\n')
            file.write(f'{mrr_head}\n')
            for hit_at_k in list_hit_at_k_head:
                file.write(f'{hit_at_k}\n')
            file.write(f'Time{repeat_no:02d} Train+Valid for {epoch + 1:03d} epoches | Dur {(time_end - time_start)/60:.2f} mins\n')
            file.write(f'Time{repeat_no:02d} Test Dur {(time_end_end - time_end) / 60:.2f} mins\n')
        finally:
            file.close()




