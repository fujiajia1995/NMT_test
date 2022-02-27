import argparse

config = argparse.Namespace()

config.predict = False
config.data_address = "/Users/jiajunfu/PycharmProjects/MyNMT/dataset/data_final.csv"
config.device = "cpu"
config.epoch = 15000
config.checkpoint_save_epoch_time = 100
config.batch_size = 100
config.embeddings_size = 300
#config.hidden_size = 300
config.checkpoint_address = "./check_point/checkpoint_model"
config.save_model_at = "./model/trained_compelete_model"
config.encode_embedding_output_size = 300 #it should be equal with pretrainned model
config.encode_rnn_hidden_size = 300
#config.encode_rnn_hidden_layers_num = 1
config.decode_embedding_output_size = 300 #it should be equath with pretainned  model
#config.decode_rnn_layers_num = 1
config.model_save_root = "./model"
config.source_pretrainned_model_address = "./pretrained_model/jawiki.word_vectors.300d.txt"
config.target_pretrained_model_address = "./pretrained_model/glove.6B.300d.txt"
config.unk_token = "<UNK>"
config.sos_token = "<SOS>"
config.eos_token = "<EOS>"
config.mask_token = "<MASK>"
config.clip = 1
config.lr = 0.03
config.dropout_p = 0.3

