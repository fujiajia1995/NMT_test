# -*- utf-8 -*-
from vocabulary import MyNMTvocabulary
from config import config
import numpy as np
import pandas
from collections import defaultdict
from tqdm import tqdm
import torch


class NMTvectorize(object):
    def __init__(self, source_vocab, target_vocab, source_max_len, target_max_len, source_glove_address, target_glove_address):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

        self.source_glove_address = source_glove_address
        self.target_glove_address = target_glove_address

    @classmethod
    def load_from_df_file(cls, NMT_df, source_glove_address=None, target_glove_address=None):
        source_vocab = MyNMTvocabulary(unk_token=config.unk_token,sos_token=config.sos_token, eos_token=config.eos_token,
                                       mask_token=config.mask_token)
        target_vocab = MyNMTvocabulary(unk_token=config.unk_token, sos_token=config.sos_token, eos_token=config.eos_token,
                                       mask_token=config.mask_token)
        source_max_len = 0.0
        target_max_len = 0.0

        for index, row in NMT_df.iterrows():
            try:
                source_tokens = row["source_sentence"].split(" ")
            except:
                print(row["source_sentence"], row["target_sentence"],index)
            if len(source_tokens) > source_max_len:
                source_max_len = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)
            try:
                target_tokens = row["target_sentence"].split(" ")
            except:
                print(row["target_sentence"], row["source_sentence"],row["index"])
            if len(target_tokens) > target_max_len:
                target_max_len = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, source_max_len, target_max_len, source_glove_address, target_glove_address)

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        if vector_length < 0:
            vector_length = len(indices)
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index
        return vector

    def _get_source_indices(self, text):
        indices = [self.source_vocab.sos_index]
        indices.extend([self.source_vocab.lookup_token(token)
                   for token in text.split(" ") if token != ""])
        indices.append(self.source_vocab.eos_index)
        return indices

    def _get_target_indices(self, text):
        indices = [self.target_vocab.lookup_token(token)
                   for token in text.split(" ") if token != ""]
        x_indices = [self.target_vocab.sos_index] + indices
        y_indices = indices + [self.target_vocab.eos_index]
        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_max_length=True):
        source_vector_length = -1
        target_vector_length = -1
        if use_max_length:
            source_vector_length = self.source_max_len+2
            target_vector_length = self.target_max_len+1

        source_indices = self._get_source_indices(source_text)
        source_len = len(source_indices)
        source_vector = self._vectorize(source_indices,
                                        vector_length=source_vector_length,
                                        mask_index=self.source_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text,)
        target_x_vector = self._vectorize(target_x_indices,
                                          vector_length=target_vector_length,
                                          mask_index=self.target_vocab.mask_index)

        target_y_vector = self._vectorize(target_y_indices,
                                          vector_length=target_vector_length,
                                          mask_index=self.target_vocab.mask_index)

        return {
            "source_vector": source_vector,
            "target_x_vector": target_x_vector,
            "target_y_vector": target_y_vector,
            "source_length": source_len
        }

    def get_embedding_wight(self, type):
        print("read pretrainen " + type + " embedding")
        embedding_dict = defaultdict(list)
        if type == "source":
            address = self.source_glove_address
            vocab = self.source_vocab
        elif type == "target":
            address = self.target_glove_address
            vocab = self.target_vocab
        else:
            raise KeyError("vectorizer type not found")
        with open(address, "r") as f:
            for i in tqdm(f):
                temo_list = i.split(' ')
                for j in range(1, len(temo_list)):
                    temo_list[j] = float(temo_list[j])
                embedding_dict[temo_list[0]].extend(temo_list[1:])
        result_list = []
        for i in tqdm(range(len(vocab))):
            #print(i)
            if vocab.lookup_index(i) in embedding_dict.keys():
                result_list.append(embedding_dict[vocab.lookup_index(i)])
            else:
                if type == "source":
                    result_list.append([float(0) for i in range(config.encode_embedding_output_size)])
                else:
                    result_list.append([float(0) for i in range(config.decode_embedding_output_size)])
        embedding_weight = torch.tensor(result_list).float().to(config.device)
        #print(embedding_weight[4])
        #print(embedding_weight.size())
        return embedding_weight





if __name__ == "__main__":
    df = pandas.read_csv(config.data_address, sep="\t")
    temp = NMTvectorize.load_from_df_file(df, config.source_pretrainned_model_address, config.target_pretrained_model_address)
    #temp.get_embedding_wight(type="target")
    result = temp.vectorize("そっち の ほう が 人 の 目 を 引く","1")
    print( torch.tensor(result["source_vector"]).unsqueeze(0) , torch.tensor([result["source_length"]]))




