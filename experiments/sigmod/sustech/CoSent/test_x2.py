import queue
import time

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from data_helper import CustomDataset, load_data, load_test_data, pad_to_maxlen
from model import Model
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert import BertTokenizer


class MyDataset(Dataset):
    def __init__(self, sentence, tokenizer):
        self.sentence = sentence
        # self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index], text_pair=None, add_special_tokens=True, return_token_type_ids=True
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {"input_id_list": input_ids, "attention_list": attention_mask, "token_list": token_type_ids}


def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len = max([len(d["input_id_list"]) for d in batch])

    if max_len > 512:
        max_len = 512

    # 定一个全局的max_len
    # max_len = 128

    input_ids, attention_mask, token_type_ids = [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item["input_id_list"], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item["attention_list"], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item["token_list"], max_len=max_len))

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


def encode(model, sentences) -> np.array:
    tokenizer = BertTokenizer.from_pretrained("./x2_model")
    dataset = MyDataset(sentence=sentences, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=256, collate_fn=collate_fn)
    trained_embedding = []
    for batch in dataloader:
        input_ids_list, attention_mask_list, _ = batch
        batch_embedding = model(
            input_ids=input_ids_list, attention_mask=attention_mask_list, encoder_type="fist-last-avg"
        )
        batch_embedding = batch_embedding.detach()
        batch_embedding = F.normalize(batch_embedding, p=2, dim=1).numpy()
        # trained_embedding=torch.cat((trained_embedding,batch_embedding),0)
        trained_embedding.extend(batch_embedding)
    return np.array(trained_embedding)


def test(model, data, limit, column):
    sentences = data[column]
    tokenizer = BertTokenizer.from_pretrained("./prajjwal1_bert-tiny")
    # input_ids_list=[]
    # attention_mask_list=[]
    # token_type_list=[]
    # for s in sentences:
    #     inputs =tokenizer.encode_plus(
    #             text=s,
    #             text_pair=None,
    #             add_special_tokens=True,
    #             return_token_type_ids=True
    #         )
    #     input_ids_list.append(inputs['input_ids'])
    #     attention_mask_list.append(inputs['attention_mask'])
    #     token_type_list.append(inputs["token_type_ids"])
    # dataset=MyDataset(sentence=sentences,tokenizer=tokenizer)
    # dataloader=DataLoader(dataset=dataset,batch_size=32,collate_fn=collate_fn)
    # input_ids_list,attention_mask_list,token_type_list=collate_fn(input_ids_list,attention_mask_list,token_type_list)
    dataset = MyDataset(sentence=sentences, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=256, collate_fn=collate_fn)
    # input_ids_list,attention_mask_list,token_type_list=collate_fn(input_ids_list,attention_mask_list,token_type_list)
    # trained_embedding=torch.Tensor()
    trained_embedding = []
    for batch in dataloader:
        input_ids_list, attention_mask_list, _ = batch
        batch_embedding = model(
            input_ids=input_ids_list, attention_mask=attention_mask_list, encoder_type="fist-last-avg"
        )
        batch_embedding = batch_embedding.detach()
        batch_embedding = F.normalize(batch_embedding, p=2, dim=1).numpy()
        # trained_embedding=torch.cat((trained_embedding,batch_embedding),0)
        trained_embedding.extend(batch_embedding)
        print(len(trained_embedding))
    # trained_embedding = model(input_ids=input_ids_list, attention_mask=attention_mask_list, encoder_type='fist-last-avg')

    numpy_embedding = np.array(trained_embedding)
    topk = 50
    index = faiss.IndexHNSWFlat(len(numpy_embedding[0]), 8)
    index.hnsw.efConstruction = 100
    index.add(numpy_embedding)
    index.hnsw.efSearch = 256
    D, I = index.search(numpy_embedding, topk)
    shape0 = len(D)
    shape1 = len(D[0])
    hashset = {}
    maxscore = 0
    pq = queue.PriorityQueue()
    for j in range(shape1):
        flag = True
        for i in range(shape0):
            s1 = min(I[i][j], i)
            s2 = max(I[i][j], i)
            token = str(s1) + " " + str(s2)
            if s1 == s2 or token in hashset:
                continue
            hashset[token] = 1
            if pq.qsize() < limit:
                flag = False
                if not True:
                    continue
                pq.put((-D[i][j], s1, s2))
                maxscore = max(maxscore, D[i][j])
            elif D[i][j] < maxscore:
                flag = False
                if not True:
                    continue
                pq.put((-D[i][j], s1, s2))
                entry = pq.get()
                maxscore = -entry[0]
        if flag:
            break
    output = []
    while not pq.empty():
        score, i, j = pq.get()
        lid = data["id"][i]
        rid = data["id"][j]
        output.append((min(lid, rid), max(lid, rid)))
        if len(output) == limit:
            break

    return output


def recall_calculation(output, gnd):
    predict = output
    cnt = 0
    for i in range(len(predict)):
        if not gnd[(gnd["lid"] == predict[i][0]) & (gnd["rid"] == predict[i][1])].empty:
            cnt += 1
    recall = cnt / gnd.values.shape[0]
    return recall


def save_output(
    X1_candidate_pairs, X2_candidate_pairs
):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for RF_dataset X1 and 2000000 pairs for RF_dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for RF_dataset X1 and 2000000 pairs for RF_dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = (
        X1_candidate_pairs + X2_candidate_pairs
    )  # make sure to have the pairs in the first RF_dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for RF_dataset X1, and the remaining pairs are for RF_dataset X2
    output_df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    mode = 0
    if mode == 0:
        model_path = "./x1_model/base_model_epoch_{}.bin".format(62)
        my_model = Model("./prajjwal1_bert-tiny/config.json", "./prajjwal1_bert-tiny/pytorch_model.bin")
        my_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        raw_data = pd.read_csv("X1.csv")
        raw_data["title"] = raw_data.title.str.lower()
        x1_pair = test(my_model, raw_data, 1000000, "title")
        raw_data = pd.read_csv("X2.csv")
        raw_data["name"] = raw_data.name.str.lower()
        # x2_pair = test(my_model, raw_data, 2000000, 'name')
        x2_pair = []
        save_output(x1_pair, x2_pair)
        print(time.time())
    elif mode == 1:
        test_data = pd.read_csv("../x2_test.csv")
        train_data = pd.read_csv("../x2_train.csv")
        origin_data = pd.read_csv("../X2.csv")
        test_gnd = pd.read_csv("../y2_test.csv")
        train_gnd = pd.read_csv("../y2_train.csv")
        origin_gnd = pd.read_csv("../Y2.csv")
        test_limit = 1357
        train_limit = 3035
        origin_limit = 4392
        my_model = Model()
        for i in range(100):
            model_path = "./outputs/base_model_epoch_{}.bin".format(i)
            my_model.load_state_dict(torch.load(model_path))
            test_pair = test(my_model, test_data, test_limit, "title")
            train_pair = test(my_model, train_data, train_limit, "title")
            origin_pair = test(my_model, origin_data, origin_limit, "name")
            print(
                "Model: %s, Test set recall: %f, Train set recall %f, Origin data recall: %f"
                % (
                    model_path,
                    recall_calculation(test_pair, test_gnd),
                    recall_calculation(train_pair, train_gnd),
                    recall_calculation(origin_pair, origin_gnd),
                )
            )
