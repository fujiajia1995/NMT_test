from torch.utils.data import DataLoader
import torch
from config import config
import pandas
from torch.nn.utils import clip_grad_norm_
from vectorizer import NMTvectorize
from dataset import NMTdataset
from nmt_model import mymodule
from torch.nn import functional as F
from tqdm import tqdm
import os


def generate_nmt_batches(dataset, batch_size, device, shuffle=True, drop_last=True):
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)

    for data_dict in dataloader:
        x_lengths = data_dict["source_length"]
        x_sorted_lengths = x_lengths.argsort().tolist()[::-1]

        result = {}
        for name, tensor in data_dict.items():
            result[name] = data_dict[name][x_sorted_lengths].to(device)
        yield result


train_stat = {
    "epoch": 0,
    "train_accu": [],
    "train_loss": [],
    "val_loss": [],
    "val_accu": [],
    "test_accu": -1,
    "test_loss": 100
}


def train(epoch_num):
    print("trainning", end="")
    for epoch_index in tqdm(range(epoch_num)):
        train_stat["epoch"] = epoch_index

        mydataset.choose("train")
        batch_train_generater = generate_nmt_batches(mydataset,
                                                     batch_size=config.batch_size,
                                                     device=config.device)

        mymodel.train()
        train_loss_old = 0
        for batch_index, batch_data in enumerate(batch_train_generater):
            optimizer.zero_grad()
            y_pred = mymodel(source_x=batch_data["source_vector"], x_lengths=batch_data["source_length"],
                             y_source=batch_data["target_x_vector"])
            target = F.one_hot(batch_data["target_y_vector"], len(vectorizer.target_vocab))
            train_loss = loss_function(y_pred, target.float())
            train_loss.backward()
            clip_grad_norm_(mymodel.parameters(), config.clip)
            optimizer.step()
            change = train_loss-train_loss_old
            train_loss_old = train_loss
        if epoch_index% config.checkpoint_save_epoch_time == 0:
            torch.save({
                "epoch": epoch_index,
                "model_state_dict": mymodel.state_dict(),
                "optimizier_state_dict": optimizer.state_dict(),
            }, config.checkpoint_address)

        dev_loss_old = 0
        mydataset.choose("dev")
        batch_dev_generator = generate_nmt_batches(mydataset,
                                                   batch_size=config.batch_size,
                                                   device=config.device)
        mymodel.eval()
        for batch_index, batch_data in enumerate(batch_dev_generator):
            y_dev_pred = mymodel(source_x=batch_data["source_vector"], x_lengths=batch_data["source_length"],
                                 y_source=batch_data["target_x_vector"])
            dev_target = F.one_hot(batch_data["target_y_vector"], len(vectorizer.target_vocab))
            dev_loss = loss_function(y_dev_pred, dev_target.float())
            dev_change = dev_loss_old - dev_loss
            dev_loss_old = dev_loss

        result = vectorizer.vectorize("そっち の ほう が 人 の 目 を 引く", "1")
        test_sentence = torch.tensor(result["source_vector"]).unsqueeze(0)
        test_lengths = torch.tensor([result["source_length"]])
        result = mymodel.predict(test_sentence, test_lengths)
        print("\n")
        for index, i in enumerate(result):
            print(vectorizer.target_vocab.lookup_index(i), end=" ")
            if index > 5:
                break
        print("\n")

        tqdm.write("train loss:" + str(float(train_loss)) +
                   " dev_loss" + str(float(dev_loss)), end="")
    torch.save(mymodel.state_dict(), config.save_model_at)


if __name__ == "__main__":
    df = pandas.read_csv(config.data_address,error_bad_lines=False, sep="\t")
    vectorizer = NMTvectorize.load_from_df_file(df,
                                                source_glove_address=config.source_pretrainned_model_address,
                                                target_glove_address=config.target_pretrained_model_address)
    if config.predict == False:
        encode_embedding_weight = vectorizer.get_embedding_wight("source")
        decode_embedding_weight = vectorizer.get_embedding_wight("target")
    else:
        encode_embedding_weight = None
        decode_embedding_weight = None
    eos_index = vectorizer.target_vocab.lookup_token(config.eos_token)
    bos_index = vectorizer.target_vocab.lookup_token(config.sos_token)
    mydataset = NMTdataset(df, vectorizer)
    mymodel = mymodule(
        enceode_word_size=len(vectorizer.source_vocab),
        encode_output_embedding=config.encode_embedding_output_size,
        target_word_size=len(vectorizer.target_vocab),
        target_output_embedding=config.decode_embedding_output_size,
        rnn_hidden_size=config.encode_rnn_hidden_size,
        target_word_mask=vectorizer.target_vocab.lookup_token(config.mask_token),
        source_word_mask=vectorizer.source_vocab.lookup_token(config.mask_token),
        encode_embedding_weight=encode_embedding_weight,
        decode_embedding_weight=decode_embedding_weight,
        eos_index=eos_index,
        bos_index=bos_index
    ).to(config.device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=config.lr)
    if not config.predict:
        if os.path.exists(config.checkpoint_address):
            """
            "epoch": epoch_index,
            "model_state_dict": mymodel.state_dict(),
            "optimizier_state_dict": optimizer.state_dict(),
            """
            checkpoint = torch.load(config.checkpoint_address,map_location=config.device)
            mymodel.load_state_dict(checkpoint["model_state_dict"]).to(config.device)
            optimizer.load_state_dict(checkpoint["optimizier_state_dict"])
            train(config.epoch-int(checkpoint["epoch"]))
        else:
            train(config.epoch)
    else:
        mymodel.load_state_dict(torch.load(config.save_model_at))
        result = vectorizer.vectorize("そっち の ほう が 人 の 目 を 引く", "1")
        test_sentence = torch.tensor(result["source_vector"]).unsqueeze(0)
        test_lengths = torch.tensor([result["source_length"]])
        result = mymodel.predict(test_sentence, test_lengths)
        print("\n")
        for index, i in enumerate(result):
            print(vectorizer.target_vocab.lookup_index(i), end=" ")
            if index > 5:
                break
        print("\n")