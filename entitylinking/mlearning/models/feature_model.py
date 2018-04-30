import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np

from entitylinking import utils
from entitylinking.mlearning.models.pytorchmodel import ELModel


class ELNet(nn.Module):

    def __init__(self, parameters):
        super(ELNet, self).__init__()
        self._p = parameters
        self._dropout = nn.Dropout(p=0.1)
        self.individual_weights = nn.Linear(len(self._p['features_linking'])
                                            + self._p["poss.emb.size"]
                                            + self._p["word.emb.size"]
                                            + self._p["word.emb.size"]
                                            , self._p['sem.layer.size'])
        self.hidden_weights = nn.Linear(self._p['sem.layer.size'], self._p['sem.layer.size'])
        self.score_weights = nn.Linear(self._p['sem.layer.size'] * 2, 1)
        self.negative_weights = nn.Linear(self._p['sem.layer.size'], 1)
        self._pos_embeddings = nn.Embedding(self._p["poss.vocab.size"], self._p["poss.emb.size"], padding_idx=0)

    def forward(self, e_x, e_sig, x, x_sig):
        e_x = e_x.long()
        x = x.float()
        x_sig = x_sig.float()
        e_sig = e_sig.float()
        choices = x.size(1)

        e_x = self._pos_embeddings(e_x)
        e_x = e_x.transpose(1, 2)
        e_x = F.adaptive_avg_pool1d(e_x, 1).view(*e_x.size()[:2])
        e_x = e_x.unsqueeze(1)
        e_x = e_x.expand(e_x.size(0), choices, e_x.size(2)).contiguous()
        e_sig = e_sig.unsqueeze(1)
        e_sig = e_sig.expand(e_sig.size(0), choices, e_sig.size(2)).contiguous()

        x = torch.cat((
            x,
            x_sig,
            e_x,
            e_sig), dim=-1)
        x = x.view(-1, x.size(-1))

        i = self.individual_weights(x)
        i = F.relu(i)
        i = self.hidden_weights(i)
        i = F.relu(i)
        i = i.view(-1, choices, i.size(-1))

        s = i.transpose(1, 2)
        s = F.adaptive_max_pool1d(s, 1)
        s = s.transpose(1, 2)

        v = s.expand_as(i)
        v = torch.cat((i, v), dim=-1)
        v = v.view(-1, v.size(-1))

        v = self._dropout(v)
        x = self.score_weights(v)
        x = x.view(-1, choices)
        # x = F.relu(x)

        z = s.view(-1,  s.size(-1))

        z = self._dropout(z)
        z = self.negative_weights(z)

        # x = torch.cat((z, x), dim=-1)

        return F.sigmoid(z.squeeze(dim=-1)), F.softmax(x, dim=-1)


class FeatureModel(ELModel):

    def __init__(self, parameters, **kwargs):
        super(FeatureModel, self).__init__(parameters=parameters, **kwargs)
        self._pos2idx = utils.create_elements_index({utils.map_pos(p) for p in utils.corenlp_pos_tagset})
        self._p["poss.vocab.size"] = len(self._pos2idx)
        if "sem.layer.size" not in parameters:
            self._p["sem.layer.size"] = 10
        if "poss.emb.size" not in parameters:
            self._p["poss.emb.size"] = 3
        self._p['features_linking'] = ["freq",
                               "lev_main_label",
                               "lev_matchedlabel",
                               "lev_sentence",
                               "match_diff",
                               "num_related_entities",
                               "num_related_relations",
                               "singature_overlap_score",
                               "mention_tokens_len"
                               ]

        self._p['features_linking_size'] = len(self._p['features_linking'])
        self._p['max.ngram.len'] = 4
        self._p['features_entity_size'] = self._p['max.ngram.len']  # for part of speech tags
        self._p['feature.size'] = self._p['features_linking_size'] + self._p['features_entity_size']

    def _get_torch_net(self):
        net = ELNet(self._p)
        return net

    def encode_data_instance(self, instance):
        entity, linkings = instance
        entity_vector = np.zeros(self._p['features_entity_size'], dtype=np.int32)
        instance_vector = np.zeros((len(linkings), self._p['features_linking_size']), dtype=np.float32)
        instance_emb_vector = np.zeros((len(linkings), self._p['word.emb.size']), dtype=np.float32)
        for t_id, p in enumerate(entity.get("poss", [])[:self._p['max.ngram.len']]):
            entity_vector[t_id] = self._pos2idx.get(utils.map_pos(p), self._pos2idx[utils.unknown_el])
        if len(entity.get("sentence_content_tokens", [])) > 0:
            entity_emb_vector = np.average([self._embedding_matrix[utils.get_word_idx(w, self._word2idx)]
                                            for w in entity.get("sentence_content_tokens", [])], axis=0)
        else:
            entity_emb_vector = np.zeros(self._p['word.emb.size'], dtype=np.float32)
        for i, l in enumerate(linkings):
            l = self._normalize_candidate_features(l)
            for j, feature_name in enumerate(self._p['features_linking']):
                instance_vector[i, j] = l.get(feature_name, 1.0) + utils.epsilon
            signature_word_ids = [utils.get_word_idx(w, self._word2idx) for w in l.get("signature", [])]
            if len(signature_word_ids) > 0:
                instance_emb_vector[i] = np.average([self._embedding_matrix[w_id] for w_id in signature_word_ids], axis=0)
        return entity_vector, entity_emb_vector, instance_vector, instance_emb_vector

    def encode_batch(self, data_without_targets, verbose=False):
        entities_data, candidates = tuple(data_without_targets)
        choices = np.max([len(el) for el in candidates])
        entities_matrix = np.zeros((len(entities_data), self._p['features_entity_size']), dtype=np.int32)
        entities_emb_matrix = np.zeros((len(entities_data), self._p['word.emb.size']))
        candidates_matrix = np.zeros((len(candidates), choices, self._p['features_linking_size']))
        candidates_emb_matrix = np.zeros((len(candidates), choices, self._p['word.emb.size']))
        for i, (entity, linkings) in enumerate(zip(entities_data, candidates)):
            entity_encoded, entity_emb_vector, linkings_encoded, linkings_emb_vector = self.encode_data_instance((entity, linkings))
            entities_matrix[i, :len(entity_encoded)] = entity_encoded
            entities_emb_matrix[i, :len(entity_emb_vector)] = entity_emb_vector
            candidates_matrix[i, :len(linkings_encoded)] = linkings_encoded
            candidates_emb_matrix[i, :len(linkings_emb_vector)] = linkings_emb_vector
        return entities_matrix, entities_emb_matrix, candidates_matrix, candidates_emb_matrix

