from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import GRU
from torch.nn import GRUCell
from torch.nn import Embedding
from torch.nn.functional import softmax
from torch.nn import Module
import torch
from config import config


def myattention(query, key, value):
    batch_size, num_vectors, feature_size = key.size()
    vector_scores = torch.matmul(key, query.view(query.size(0), -1, 1))
    vector_scores = vector_scores.view(batch_size, -1)
    vector_scores = softmax(vector_scores, dim=-1)
    result_vector = torch.matmul(vector_scores.view(vector_scores.size(0), 1, -1), value)
    result_vector = result_vector.view(result_vector.size(0), -1)
    return result_vector


class NMTEncoder(Module):
    def __init__(self, input_embedding_size, output_embedding_size, rnn_hidden_size, voc_mask, embedding_weight=None):
        super(NMTEncoder, self).__init__()
        self.embedding_drop_out = torch.nn.Dropout(config.dropout_p)
        self.embedding_weight = embedding_weight
        self.source_embedding = Embedding(input_embedding_size, output_embedding_size, padding_idx=voc_mask,
                                          _weight=self.embedding_weight)
        self.lstm = GRU(input_size=output_embedding_size, hidden_size=rnn_hidden_size, bidirectional=True)

    def forward(self, x, x_lengths):
        x = self.embedding_drop_out(self.source_embedding(x))
        x_packed = pack_padded_sequence(x, x_lengths, batch_first=True)
        encoder_hidden_sequence, encoder_last_hidden = self.lstm(x_packed)
        encoder_hidden_sequence, _ = pad_packed_sequence(encoder_hidden_sequence, batch_first=True)
        encoder_last_hidden = encoder_last_hidden.permute(1, 0, 2)
        encoder_last_hidden = encoder_last_hidden.contiguous().view(encoder_last_hidden.size(0), -1)
        return encoder_hidden_sequence, encoder_last_hidden


class NMTDecoder(Module):
    def __init__(self, input_embedding, output_embedding, rnn_hidden_size, mask_index, embedding_weight,bos_index,eos_index):
        super(NMTDecoder, self).__init__()
        self.eos_index = eos_index
        self.bos_index = bos_index
        self.embedding_weight = embedding_weight
        self.embedding_drop_out = torch.nn.Dropout(config.dropout_p)
        self.attention_drop_out = torch.nn.Dropout(config.dropout_p)
        self.classifier_drop_out = torch.nn.Dropout(config.dropout_p)
        self._hidden_size = rnn_hidden_size
        self.hidden_map = torch.nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.embedding = torch.nn.Embedding(input_embedding, output_embedding, _weight=self.embedding_weight)
        self.cell = GRUCell(input_size=output_embedding+rnn_hidden_size, hidden_size=rnn_hidden_size)
        self.classifier = torch.nn.Linear(rnn_hidden_size*2, input_embedding)
        self.mask_index = mask_index

    def _init_context_vector(self, batch_size):
        return torch.zeros(batch_size, self._hidden_size)

    def forward(self, encode_hidden_sequence, encode_last_hidden, y_source):
        h_t = self.hidden_map(encode_last_hidden).to(config.device)
        batch_size = encode_hidden_sequence.size(0)
        context_vector = self._init_context_vector(batch_size).to(config.device)
        y_source = y_source.permute(1, 0)
        result = []
        for sequence_index in range(y_source.size(0)):
            now_y_source = self.embedding_drop_out(self.embedding(y_source[sequence_index]))
            rnn_input = torch.cat([now_y_source, context_vector], dim=-1)
            h_t = self.cell(rnn_input, h_t)
            context_vector = self.attention_drop_out(myattention(h_t, encode_hidden_sequence, encode_hidden_sequence))
            prediction = torch.cat([context_vector, h_t], dim=-1)
            score_for_y_t_index = self.classifier_drop_out(self.classifier(prediction))
            #score_for_y_t_index = softmax(score_for_y_t_index, dim=-1)
            result.append(score_for_y_t_index)
        result = torch.stack(result)
        result = result.permute(1, 0, 2)
        return result

    def predict(self,encode_hidden_sequence, encode_last_hidden):
        encode_hidden_sequence = encode_hidden_sequence
        encode_last_hidden = encode_last_hidden
        h_t = self.hidden_map(encode_last_hidden).to(config.device)
        batch_size = 1
        context_vector = self._init_context_vector(batch_size)
        context_vector = context_vector
        y_source = torch.tensor([self.bos_index])
        result = []
        for i in range(30):
            #print(y_source)
            now_y_source = y_source
            embedding_y = self.embedding(now_y_source)
            #print(embedding_y.size(), context_vector.size())
            rnn_input = torch.cat([embedding_y, context_vector], dim=-1)
            h_t = self.cell(rnn_input, h_t)
            context_vector = myattention(h_t, encode_hidden_sequence, encode_hidden_sequence)
            prediction = torch.cat([context_vector, h_t], dim=-1)
            score_for_y_index = self.classifier(prediction)
            predicted_index = torch.argmax(score_for_y_index, dim=-1)
            result.append(int(predicted_index))
            y_source = torch.tensor([predicted_index])
            if y_source == self.eos_index:
                break
        return result


class mymodule(Module):
    def __init__(self,enceode_word_size, encode_output_embedding,
                 target_word_size, target_output_embedding,
                 rnn_hidden_size, target_word_mask, source_word_mask,
                 eos_index,bos_index,
                 encode_embedding_weight=None, decode_embedding_weight=None
                 ):
        super(mymodule, self).__init__()
        self.encoder = NMTEncoder(input_embedding_size=enceode_word_size,output_embedding_size=encode_output_embedding,
                                  rnn_hidden_size=rnn_hidden_size, voc_mask=source_word_mask,
                                  embedding_weight=encode_embedding_weight)
        decode_rnn_hidden_size = rnn_hidden_size*2
        self.decoder = NMTDecoder(input_embedding=target_word_size, output_embedding=target_output_embedding,
                                  rnn_hidden_size=decode_rnn_hidden_size, mask_index=target_word_mask,
                                  embedding_weight=decode_embedding_weight,eos_index=eos_index,bos_index=bos_index)

    def forward(self, source_x, x_lengths, y_source):
        encode_output_stat, encode_last_hidden_stat = self.encoder(source_x, x_lengths)
        result = self.decoder(encode_hidden_sequence=encode_output_stat, encode_last_hidden=encode_last_hidden_stat,
                              y_source= y_source)
        return result

    def predict(self, source_x, x_lengths):
        encode_output_stat, encode_last_hidden_stat = self.encoder(source_x, x_lengths)
        result = self.decoder.predict(encode_output_stat, encode_last_hidden_stat)
        return result





if __name__ == "__main__":
    x_source = torch.randint(1, 10, (10, 20))
    x_lengths = torch.randint(20, 21, (10,))
    target_x_vector = torch.randint(1, 10, (10, 27))
    MyModule = mymodule(enceode_word_size=10, encode_output_embedding=10, target_word_size=10,
                        target_output_embedding=10, rnn_hidden_size=10, target_word_mask=0, source_word_mask=0,
                        eos_index=4,bos_index=1)
    print(MyModule(x_source, x_lengths, target_x_vector).size())
    x_source = torch.randint(1, 10, (1, 20))
    x_lengths = torch.randint(20, 21, (1,))
    #target_x_vector = torch.randint(1, 10, (10, 27))
    MyModule.predict(x_source, x_lengths)