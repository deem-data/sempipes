"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-07
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, T5Config


class Model(nn.Module):
    def __init__(self, config_path, bert_path):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(config_path)
        self.bert = BertModel.from_pretrained(bert_path, config=self.config)

    def forward(self, input_ids, attention_mask, encoder_type="fist-last-avg"):
        """
        :param input_ids:
        :param attention_mask:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        """
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)

        if encoder_type == "fist-last-avg":
            # 第一层和最后一层的隐层取出  然后经过平均池化
            first = output.hidden_states[
                1
            ]  # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)  # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2
            ).squeeze(-1)
            # print(final_encoding)
            return final_encoding

        if encoder_type == "last-avg":
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output


class myTokenizer:
    def __init__(self, tokenizer_path: str) -> None:
        self.config = AutoConfig.from_pretrained(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_length = 512

    def tokenize(self, texts):
        output = {}
        to_tokenize = [texts]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        output = self.tokenizer(
            *to_tokenize, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_seq_length
        )
        return output


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


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


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


#     # 按batch进行padding获取当前batch中最大长度
#
#
#     # 定一个全局的max_len
#
#
#


def encode(model, sentences, tokenizer: myTokenizer) -> np.array:
    # length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    # sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    sentences_sorted = sentences
    # dataset = MyDataset(sentence=sentences_sorted, tokenizer=tokenizer)
    # dataloader = DataLoader(dataset=dataset, batch_size=256, collate_fn=collate_fn)
    trained_embedding = []
    # from tqdm.autonotebook import trange
    batch_size = 256
    # for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=True):
    #     sentences_batch = sentences_sorted[start_index:start_index + batch_size]
    #     # inputs = tokenizer.encode_plus(
    #     #     text=sentences_batch,
    #     #     text_pair=None,
    #     #     add_special_tokens=True,
    #     #     return_token_type_ids=True
    #     # )
    #     inputs=tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences_batch,return_token_type_ids=False,return_attention_mask=True)
    #     input_ids = inputs['input_ids']
    #     attention_mask = inputs['attention_mask']
    #     #token_type_ids = inputs["token_type_ids"]
    #     input_ids_list, attention_mask_list=new_collate_fn(input_ids,attention_mask)
    #     batch_embedding = model(input_ids=input_ids_list, attention_mask=attention_mask_list,
    #                             encoder_type='fist-last-avg')
    #     batch_embedding = batch_embedding.detach()
    #     batch_embedding = torch.nn.functional.normalize(batch_embedding, p=2, dim=1).numpy()
    #     trained_embedding.extend(batch_embedding)

    # for batch in dataloader:
    #     input_ids_list, attention_mask_list, _ = batch
    #     batch_embedding = model(input_ids=input_ids_list, attention_mask=attention_mask_list,
    #                             encoder_type='fist-last-avg')
    #     batch_embedding = batch_embedding.detach()
    #     batch_embedding = torch.nn.functional.normalize(batch_embedding, p=2, dim=1).numpy()
    #     trained_embedding.extend(batch_embedding)

    for start_index in range(0, len(sentences_sorted), batch_size):
        sentence_batch = sentences_sorted[start_index : start_index + batch_size]
        features = tokenizer.tokenize(texts=sentence_batch)
        input_ids_list = features["input_ids"]
        attention_mask_list = features["attention_mask"]
        batch_embedding = model(
            input_ids=input_ids_list, attention_mask=attention_mask_list, encoder_type="fist-last-avg"
        )
        batch_embedding = batch_embedding.detach()
        batch_embedding = F.normalize(batch_embedding, p=2, dim=1).cpu()
        trained_embedding.extend(batch_embedding)

    # all_embeddings = [trained_embedding[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = np.asarray([emb.numpy() for emb in trained_embedding])
    return np.array(all_embeddings)
