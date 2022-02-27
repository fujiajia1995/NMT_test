from torch.utils.data import Dataset
import pandas
from vectorizer import NMTvectorize
from config import config
from torch.utils.data import DataLoader


class NMTdataset(Dataset):
    def __init__(self, df, vectorizer):
        self.data_df = df
        self._vectorizer = vectorizer

        self.train_df = self.data_df[self.data_df.split_kind == "train"]
        self.train_size = len(self.train_df)

        self.dev_df = self.data_df[self.data_df.split_kind == "dev"]
        self.dev_size = len(self.dev_df)

        self.test_df = self.data_df[self.data_df.split_kind == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dataset = {
            "train": (self.train_df, self.train_size),
            "dev": (self.dev_df, self.dev_size),
            "test": (self.test_df, self.test_size)
        }
        #self.choose("train")

    def _get_vectorizer(self):
        return self._vectorizer

    def choose(self, choice="train"):
        self._target_df,self._target_size = self._lookup_dataset[choice]

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        result \
            = self._vectorizer.vectorize(row.source_sentence, row.target_sentence, True)
        #print(row.source_sentence,row.target_sentence)
        return {
            "source_vector": result["source_vector"],
            "target_x_vector": result["target_x_vector"],
            "target_y_vector": result["target_y_vector"],
            "source_length": result["source_length"]
        }

    def __len__(self):
        return self._target_size

    def get_num_batches(self, batch_size):
        return self._target_size // batch_size


if __name__ == "__main__":
    def generate_nmt_batches(dataset, batch_size, device, shuffle=True, drop_last=True):
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last)

        for data_dict in dataloader:
            x_lengths = data_dict["source_length"]
            # print(x_lengths)
            x_sorted_lengths = x_lengths.argsort().tolist()[::-1]

            result = {}
            for name, tensor in data_dict.items():
                result[name] = data_dict[name][x_sorted_lengths].to(device)
            yield result
    df = pandas.read_csv(config.data_address,error_bad_lines=False, sep="\t")
    vectorizer = NMTvectorize.load_from_df_file(df)
    mydataset = NMTdataset(df, vectorizer)
    mydataset.choose("test")
    print(len(mydataset))
    print(mydataset[2])
    dataload = generate_nmt_batches(mydataset, batch_size=config.batch_size, device=config.device)
    for batch_index, batch_data in enumerate(dataload):
        print(batch_data)

