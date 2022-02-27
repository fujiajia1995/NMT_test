# -*- utf-8 -*-


class MyNMTvocabulary(object):
    def __init__(self, unk_token, sos_token, eos_token, mask_token, token_to_idx=None, add_unk=True,):
        if token_to_idx == None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._mask_token = mask_token
        self._unk_token = unk_token
        self._sos_token = sos_token
        self._eos_token = eos_token
        self.unk_index = -1
        self.mask_index = self.add_token(mask_token)
        if self._add_unk:
            self.unk_index = self.add_token(unk_token)
        self.sos_index = self.add_token(sos_token)
        self.eos_index = self.add_token(eos_token)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        if self._add_unk:
            return  self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("cant find index")
        return self._idx_to_token[index]

    def __len__(self):
        return len(self._idx_to_token)