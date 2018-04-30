import string

import numpy as np
from collections import defaultdict

from entitylinking import utils

import torch
from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from entitylinking.mlearning.models.pytorchmodel import ELModel


class Words2VectorNet(nn.Module):

    def __init__(self, parameters, word_embeddings=None):
        super(Words2VectorNet, self).__init__()
        self._p = parameters
        self._dropout = nn.Dropout(p=self._p.get('dropout', '0.1'))
        self._word_embedding = nn.Embedding(self._p['word.vocab.size'], self._p['word.emb.size'], padding_idx=0)
        if word_embeddings is not None:
            word_embeddings = torch.from_numpy(word_embeddings).float()
            self._word_embedding.weight = nn.Parameter(word_embeddings)
        self._word_embedding.weight.requires_grad = False

        self._pos_embedding = nn.Embedding(3, self._p['poss.emb.size'], padding_idx=0)

        self._word_encoding_conv = nn.Conv1d(self._p['word.emb.size'] + self._p['poss.emb.size'],
                                             self._p['word.conv.size'],
                                             self._p['word.conv.width'],
                                             padding=self._p['word.conv.width']//2)

        self._nonlinearity = nn.ReLU() if self._p.get('enc.activation', 'tanh') == 'relu' else nn.Tanh()
        self._convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self._p['word.conv.size'],
                                    out_channels=self._p['word.conv.size'],
                                    kernel_size=self._p['word.conv.width'],
                                    padding=self._p['char.conv.width']//2 * 2**(j + 1) if not self._p.get("legacy.mode", False) else self._p['char.conv.width']//2 + 2**(j + 1),
                                    dilation=2**(j + 1),
                                    bias=True),
                          self._nonlinearity
                          )
            for j in range(self._p.get('word.conv.depth', 1))
        ])
        self._block_conv = nn.Conv1d(self._p['word.conv.size'],
                                     self._p['word.conv.size'],
                                     self._p['word.conv.width'],
                                     padding=self._p['word.conv.width']//2)

        self.sem_layers = nn.Sequential(
            self._dropout,
            nn.Linear(self._p['word.conv.size'], self._p['word.enc.size']),
            self._nonlinearity,
        )

        self._pool = nn.AdaptiveMaxPool1d(1) if self._p.get('enc.pooling', 'max') == 'max' else nn.AdaptiveAvgPool1d(1)

    def forward(self, sent_m_with_pos):
        sent_m_with_pos = sent_m_with_pos.long()
        sent_m = sent_m_with_pos[..., 0]
        positions = sent_m_with_pos[..., 1]
        sent_m = self._word_embedding(sent_m)
        positions = self._pos_embedding(positions)

        sent_m = torch.cat((sent_m, positions), dim=-1).transpose(-2, -1).contiguous()
        sent_m = self._dropout(sent_m)
        sent_m = self._word_encoding_conv(sent_m)
        sent_m = self._nonlinearity(sent_m)

        for _ in range(self._p.get("word.repeat.convs", 3)):
            for convlayer in self._convs:
                sent_m = convlayer(sent_m)
            sent_m = self._block_conv(sent_m)
            sent_m = self._nonlinearity(sent_m)

        sent_m = self._pool(sent_m).squeeze(dim=-1)
        sent_m = self.sem_layers(sent_m)
        return sent_m


class Chars2VectorNet(nn.Module):

    def __init__(self, parameters):
        super(Chars2VectorNet, self).__init__()
        self._p = parameters
        self._p['char.enc.size'] = self._p['char.conv.size'] // 2
        self._dropout = nn.Dropout(p=self._p.get('dropout', '0.1'))
        self._char_embedding = nn.Embedding(self._p["char.vocab.size"], self._p["char.emb.size"], padding_idx=0)
        self._pos_embedding = nn.Embedding(26, self._p['poss.emb.size'], padding_idx=0)

        self._char_encoding_conv = nn.Conv1d(in_channels=self._p['char.emb.size'] + self._p['poss.emb.size'],
                                             out_channels=self._p['char.conv.size'],
                                             kernel_size=self._p['char.conv.width'],
                                             padding=self._p['char.conv.width']//2)
        if not self._p.get("legacy.mode", False):
            self._char_weight_conv = nn.Conv1d(in_channels=self._p['char.emb.size'] + self._p['poss.emb.size'],
                                               out_channels=1,
                                               kernel_size=self._p['char.conv.width'],
                                               padding=self._p['char.conv.width']//2)
        self._nonlinearity = nn.ReLU() if self._p.get('enc.activation', 'tanh') == 'relu' else nn.Tanh()

        self._convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self._p['char.conv.size'],
                                    out_channels=self._p['char.conv.size'],
                                    kernel_size=self._p['char.conv.width'],
                                    padding=self._p['char.conv.width']//2 * 2**(j + 1) if not self._p.get("legacy.mode", False) else self._p['char.conv.width']//2 + 2**(j + 1),
                                    dilation=2**(j + 1),
                                    bias=True),
                          self._nonlinearity
                          )
            for j in range(self._p.get('char.conv.depth', 1))
        ])
        self._block_conv = nn.Conv1d(in_channels=self._p['char.conv.size'],
                                     out_channels=self._p['char.conv.size'],
                                     kernel_size=self._p['char.conv.width'],
                                     padding=self._p['char.conv.width']//2)

        self.sem_layers = nn.Sequential(
            self._dropout,
            nn.Linear(self._p['char.conv.size'], self._p['char.enc.size']),
            self._nonlinearity,
            )

        self._pool = nn.AdaptiveMaxPool1d(1) if self._p.get('enc.pooling', 'max') == 'max' else nn.AdaptiveAvgPool1d(1)

    def forward(self, string_m_with_pos):
        string_m_with_pos = string_m_with_pos.long()
        string_m = string_m_with_pos[..., 0]
        positions = string_m_with_pos[..., 1]
        string_m = self._char_embedding(string_m)
        positions = self._pos_embedding(positions)

        string_m = torch.cat((string_m, positions), dim=-1).transpose(-2, -1).contiguous()
        string_m = self._dropout(string_m)
        string_char_conv = self._char_encoding_conv(string_m)
        string_char_conv = self._nonlinearity(string_char_conv)

        for _ in range(self._p.get("char.repeat.convs", 3)):
            for convlayer in self._convs:
                string_char_conv = convlayer(string_char_conv)
            string_char_conv = self._block_conv(string_char_conv)
            string_char_conv = self._nonlinearity(string_char_conv)

        if self._p.get("legacy.mode", False):
            string_char_conv = self._pool(string_char_conv).squeeze(dim=-1)
        else:
            string_m_weights = self._char_weight_conv(string_m).transpose(-2, -1)
            string_m_weights = F.softmax(string_m_weights, dim=1)
            string_char_conv = torch.bmm(string_char_conv, string_m_weights).squeeze(dim=-1)
        string_char_conv = self.sem_layers(string_char_conv)

        return string_char_conv


class VectorNet(nn.Module):

    def __init__(self, parameters, word_embeddings, entity_embeddings, relation_embeddings):
        super(VectorNet, self).__init__()
        self._p = parameters
        self._p['char.enc.size'] = self._p['char.conv.size'] // 2
        self._p['word.enc.size'] = self._p['word.conv.size'] // 2
        self._dropout = nn.Dropout(p=self._p.get('dropout', '0.1'))

        self._words2vector = Words2VectorNet(self._p, word_embeddings)  # type: nn.Module
        self._chars2vector = Chars2VectorNet(self._p)  # type: nn.Module

        self._rel_embedding = nn.Embedding(self._p["relation.vocab.size"], self._p["relation.emb.size"], padding_idx=0)
        if relation_embeddings is not None:
            relation_embeddings = torch.from_numpy(relation_embeddings).float()
            self._rel_embedding.weight = nn.Parameter(relation_embeddings)
        self._rel_embedding.weight.requires_grad = False

        self._entity_embedding = nn.Embedding(self._p["entity.vocab.size"], self._p["entity.emb.size"], padding_idx=0)
        if entity_embeddings is not None:
            entity_embeddings = torch.from_numpy(entity_embeddings).float()
            self._entity_embedding.weight = nn.Parameter(entity_embeddings)
        self._entity_embedding.weight.requires_grad = False

        self._nonlinearity = nn.ReLU() if self._p.get('enc.activation', 'tanh') == 'relu' else nn.Tanh()
        self._pool = nn.AdaptiveMaxPool1d(1) if self._p.get('enc.pooling', 'max') == 'max' else nn.AdaptiveAvgPool1d(1)

        self._signature_layer = nn.Sequential(
            self._dropout,
            nn.Linear(self._p['word.emb.size'], self._p['relation.layer.size']),
            self._nonlinearity,
        )
        self._relations_layer = nn.Sequential(
            self._dropout,
            nn.Linear(self._p['relation.emb.size'], self._p['relation.layer.size']),
            self._nonlinearity,
        )
        self._entity_layer = nn.Sequential(
            self._dropout,
            nn.Linear(self._p['entity.emb.size'], self._p['entity.layer.size']),
            self._nonlinearity,
        )

        self.sem_layers = nn.Sequential(
            nn.Linear(self._p['word.enc.size']
                      + self._p['char.enc.size'] * 2
                      + self._p['entity.layer.size'] * 2
                      + self._p["relation.layer.size"] * 2
                      + len(self._p['features_linking'])
                      , self._p['sem.layer.size']),
            self._nonlinearity,
            self._dropout,
            nn.Linear(self._p['sem.layer.size'], self._p['sem.layer.size'] // 2),
            self._nonlinearity,
            *((nn.Linear(self._p['sem.layer.size'] // 2, self._p['sem.layer.size'] // 2),
               self._nonlinearity) if not self._p.get("legacy.mode", False) else ())
        )
        self.negative_layers = nn.Sequential(
            nn.Linear(self._p['word.enc.size'] * 1
                      + self._p['char.enc.size'] * 1, self._p['neg.layer.size']),
            self._nonlinearity,
            self._dropout,
            nn.Linear(self._p['neg.layer.size'], self._p['neg.layer.size'] // 2),
            self._nonlinearity,
            *((nn.Linear(self._p['neg.layer.size'] // 2, self._p['neg.layer.size'] // 2),
               self._nonlinearity) if not self._p.get("legacy.mode", False) else ())
        )

        self.score_weights = nn.Linear(self._p['sem.layer.size'], 1)
        self.negative_weight = nn.Linear(self._p['sem.layer.size'] // 2 + self._p['neg.layer.size'] // 2 + 1 + 1, 1)

    def compute_candidate_vectors(self, relations_m, relations_words_m,
                                  entities_m, candidates_m, candidates_labels_m):
        batch_signature_size = relations_m.size(-1)

        relations_words_m = relations_words_m.long()
        relations_embs = self._words2vector._word_embedding(relations_words_m).transpose(-2, -1).contiguous()
        relations_embs = self._pool(relations_embs).squeeze(dim=-1)
        relations_embs = self._signature_layer(relations_embs)
        relations_embs = relations_embs.view(-1, batch_signature_size, relations_embs.size(-1)).transpose(-2, -1)
        relations_embs_pooled = self._pool(relations_embs).squeeze(dim=-1)

        relations_m = relations_m.long()
        relations_m = self._rel_embedding(relations_m)
        relations_m = relations_m.view(-1, relations_m.size(-1))
        relations_m = self._relations_layer(relations_m)
        relations_m = relations_m.view(-1, batch_signature_size, relations_m.size(-1)).transpose(-2, -1).contiguous()
        relations_m_pooled = self._pool(relations_m).squeeze(dim=-1)

        entities_m = entities_m.long()
        entities_m = self._entity_embedding(entities_m)
        entities_m = entities_m.view(-1, entities_m.size(-1))
        entities_m = self._entity_layer(entities_m)
        entities_m = entities_m.view(-1, batch_signature_size, entities_m.size(-1)).transpose(-2, -1).contiguous()
        entities_m_pooled = self._pool(entities_m).squeeze(dim=-1)

        candidates_m = candidates_m.long()
        candidates_m = self._entity_embedding(candidates_m)
        candidates_m = candidates_m.view(-1, candidates_m.size(-1))
        candidates_m = self._entity_layer(candidates_m)

        candidates_label_emb = self._chars2vector(candidates_labels_m)

        concatenated_vectors = torch.cat((relations_m_pooled,
                                          entities_m_pooled,
                                          relations_embs_pooled,
                                          candidates_m,
                                          candidates_label_emb), dim=-1).contiguous()

        return concatenated_vectors

    def forward(self, sent_m, mention_m,
                relations_m, relations_words_m,
                entities_m, candidates_m, candidates_labels_m, features_m):
        choices = candidates_labels_m.size(1)  # number of possible candidates per one mention
        real_choices_num = torch.sum((candidates_m > 0).float(), dim=1).unsqueeze(1)

        sent_emb = self._words2vector(sent_m).unsqueeze(1)
        mention_emb = self._chars2vector(mention_m).unsqueeze(1)

        sent_emb_expanded = sent_emb.expand(sent_emb.size(0), choices, sent_emb.size(2)).contiguous()
        mention_emb_expanded = mention_emb.expand(mention_emb.size(0), choices, mention_emb.size(2)).contiguous()

        relations_words_m = relations_words_m.view(relations_words_m.size(0) *
                                                   relations_words_m.size(1) * relations_words_m.size(2), -1)
        relations_m = relations_m.view(relations_m.size(0) * relations_m.size(1), -1)
        entities_m = entities_m.view(entities_m.size(0) * entities_m.size(1), -1)
        candidates_labels_m = candidates_labels_m.view(candidates_labels_m.size(0) * candidates_labels_m.size(1), -1, 2)

        candidate_vectors = self.compute_candidate_vectors(relations_m, relations_words_m,
                                           entities_m, candidates_m, candidates_labels_m)
        candidate_vectors = candidate_vectors.view(-1, choices, candidate_vectors.size(-1))

        features_m = features_m.float()

        concatenated_embed = torch.cat((sent_emb_expanded,
                                        mention_emb_expanded,
                                        candidate_vectors,
                                        features_m
                                        ), dim=-1).contiguous()
        concatenated_embed = concatenated_embed.view(-1, concatenated_embed.size(-1))
        sem_vector = self.sem_layers(concatenated_embed)
        sem_vector = sem_vector.view(-1, choices, sem_vector.size(-1))

        sem_vector_pooled_over_choices = sem_vector.transpose(-2, -1)
        sem_vector_pooled_over_choices = self._pool(sem_vector_pooled_over_choices)
        sem_vector_pooled_over_choices = sem_vector_pooled_over_choices.transpose(-2, -1)
        sem_vector = torch.cat((sem_vector, sem_vector_pooled_over_choices.expand_as(sem_vector)), dim=-1)
        sem_vector = sem_vector.view(-1, sem_vector.size(-1))

        candidate_scores = self.score_weights(sem_vector).squeeze(dim=-1)
        candidate_scores = candidate_scores.view(-1, choices)

        negative_vector = torch.cat((sent_emb.squeeze(dim=1),
                                     mention_emb.squeeze(dim=1)), dim=1)
        negative_vector = self.negative_layers(negative_vector)
        real_choices_num = self._nonlinearity(real_choices_num)
        choices_pooled_for_negative = sem_vector_pooled_over_choices.squeeze(dim=1)
        negative_vector = torch.cat((negative_vector,
                                     choices_pooled_for_negative,
                                     F.adaptive_max_pool1d(candidate_scores.unsqueeze(1), 1).squeeze(dim=-1),
                                     real_choices_num
                                     ), dim=-1)
        negative_score = self.negative_weight(negative_vector)

        candidate_scores = self._nonlinearity(candidate_scores)
        return F.sigmoid(negative_score.squeeze(dim=-1)), candidate_scores


class VectorModel(ELModel):

    def __init__(self, parameters, **kwargs):
        super(VectorModel, self).__init__(parameters=parameters, **kwargs)
        if "sem.layer.size" not in parameters:
            self._p["sem.layer.size"] = 10
        self._p['max.ngram.len'] = 4

        self._char2idx = utils.create_elements_index(set(string.printable))
        self._p['char.vocab.size'] = len(self._char2idx)

        self._p['relation.vocab.size'] = 0
        self._rels_embedding_matrix = None
        self._rel2idx = {}
        self._p['relation.emb.size'] = 50

        self._p['entity.vocab.size'] = 0
        self._entities_embedding_matrix = None
        self._entity2idx = {}
        self._p['entity.emb.size'] = 50

        self._p['features_linking'] = ["num_related_entities",
                                       "num_related_relations",
                                       "singature_overlap_score"]

    def prepare_model(self, entities_embedding_matrix=None, entity2idx=None, rels_embedding_matrix=None, rel2idx=None,
                      **kwargs):
        if entities_embedding_matrix is None and entity2idx is None:
            self.logger.debug("Loading KB embeddings")
            self._entities_embedding_matrix, self._entity2idx, self._rels_embedding_matrix, self._rel2idx = \
                utils.load_kb_embeddings(self._p['kb.embeddings'])
        else:
            self._entities_embedding_matrix, self._entity2idx = entities_embedding_matrix, entity2idx
            self._rels_embedding_matrix, self._rel2idx = rels_embedding_matrix, rel2idx
        self.logger.info("Loaded KB embeddings: {}, {}".format(self._entities_embedding_matrix.shape,
                                                               self._rels_embedding_matrix.shape))

        self._p['relation.emb.size'] = self._rels_embedding_matrix.shape[1]
        self._p['relation.vocab.size'] = len(self._rel2idx)
        self._p['entity.emb.size'] = self._entities_embedding_matrix.shape[1]
        self._p['entity.vocab.size'] = len(self._entity2idx)

        super(VectorModel, self).prepare_model(**kwargs)
        if 'pretrained_components' in self._p:
            if 'chars2vector' in self._p['pretrained_components']:
                self._model._chars2vector.load_state_dict(
                    torch.load(self._p['pretrained_components']['chars2vector'])
                )
                self.logger.info("Pre-trained char model loaded.")
                self._model._chars2vector._char_embedding.weight.requires_grad = False
                self._model._chars2vector._pos_embedding.weight.requires_grad = False

    def load_from_file(self, path_to_model):
        """
        Load a model from file.

        :param path_to_model: path to the model file.
        """
        super(VectorModel, self).load_from_file(path_to_model)
        if self._entities_embedding_matrix is None:
            self.logger.debug("Loading KB embeddings")
            self._entities_embedding_matrix, self._entity2idx, self._rels_embedding_matrix, self._rel2idx = \
                utils.load_kb_embeddings(self._p['kb.embeddings'])
            self.logger.info("Loaded KB embeddings: {}, {}".format(self._entities_embedding_matrix.shape,
                                                                   self._rels_embedding_matrix.shape))

        self._p['relation.emb.size'] = self._rels_embedding_matrix.shape[1]
        self._p['relation.vocab.size'] = len(self._rel2idx)
        self._p['entity.emb.size'] = self._entities_embedding_matrix.shape[1]
        self._p['entity.vocab.size'] = len(self._entity2idx)

    def encode_sentence(self, tokens, mention_token_ids=None, out=None):
        tokens = ['<'] + tokens + ['>']
        if out is None:
            out = np.zeros((len(tokens), 2), dtype='int32')
        mention_token_ids = {j+1 for j in mention_token_ids} if mention_token_ids is not None else set()

        for i, token in enumerate(tokens[:out.shape[0]]):
            token = token.lower()
            out[i, 0] = utils.get_word_idx(token, self._word2idx)
            out[i, 1] = 2 if i in mention_token_ids else 1
        return out

    def encode_tokens(self, tokens, out=None):
        tokens = ['<'] + tokens + ['>']
        if out is None:
            out = np.zeros((len(tokens), 2), dtype='int32')
        for i, token in enumerate(tokens[:out.shape[0]]):
            token = token.lower()
            out[i] = utils.get_word_idx(token, self._word2idx)
        return out

    def encode_mention(self, mention, out=None):
        mention = '<' + mention + '>'
        if out is None:
            out = np.zeros((len(mention), 2), dtype='int32')
        mention = mention.strip()
        chars = [self._char2idx.get(c, self._char2idx[utils.unknown_el]) for c in mention][:out.shape[0]]
        out[:len(chars), 0] = chars
        out[:len(chars), 1] = list(range(1, len(chars) + 1))
        return out

    def encode_batch(self, data_without_targets, verbose=False):
        entities_data, candidates_data = tuple(data_without_targets)
        batch_size = len(entities_data)
        batch_candidates_per_entity = max(max(len(linkings) for linkings in candidates_data), 1)
        # batch_sent_length = max(max(len(e['sentence_tokens']) for e in entities_data), 1)
        batch_sent_length = 15
        # batch_mention_length = max(max(len(e['mention_context']) for e in entities_data), 1)
        batch_mention_length = 25
        # batch_label_length = max(max(len(l['label_tokens']) for linkings in candidates_data for l in linkings), 1)
        batch_label_length = 5
        # batch_signature_size = max(max(l['num_related_relations'] for linkings in candidates_data for l in linkings), 1)
        batch_signature_size = 25

        sentences_matrix = np.zeros((batch_size,
                                     batch_sent_length,
                                     2),
                                    dtype="int32")

        mention_matrix = np.zeros((batch_size,
                                   batch_mention_length,
                                   2),
                                  dtype="int32")
        candidate_relations_matrix = np.zeros((batch_size,
                                               batch_candidates_per_entity,
                                               batch_signature_size),
                                              dtype="int32")
        candidate_relations_words_matrix = np.zeros((batch_size,
                                                     batch_candidates_per_entity,
                                                     batch_signature_size,
                                                     batch_label_length),
                                                    dtype="int32")
        candidate_entities_matrix = np.zeros((batch_size,
                                              batch_candidates_per_entity,
                                              batch_signature_size),
                                             dtype="int32")

        candidate_matrix = np.zeros((batch_size,
                                     batch_candidates_per_entity),
                                    dtype="int32")
        candidate_label_matrix = np.zeros((batch_size,
                                           batch_candidates_per_entity,
                                           batch_mention_length,
                                           2),
                                          dtype="int32")

        features_size = len(self._p['features_linking'])
        candidate_features = np.zeros((batch_size,
                                       batch_candidates_per_entity,
                                       features_size))
        for i, (entity, linkings) in enumerate(tqdm(zip(entities_data, candidates_data),
                                                        ascii=True, ncols=100, disable=(not verbose))):
            self.encode_sentence(entity['sentence_tokens'],
                                 mention_token_ids=entity['token_ids'], out=sentences_matrix[i])
            self.encode_mention(entity['mention_context'], out=mention_matrix[i])
            for j, l in enumerate(linkings):
                entities_by_rel = defaultdict(list)
                for e in l['related_entities']:
                    entities_by_rel[e[2]].append(e)
                relations = [r for r in l['related_relations'] if r[1] in self._rel2idx]
                for k, r in enumerate(relations[:batch_signature_size]):
                    candidate_relations_matrix[i, j, k] = self._rel2idx.get(r[1], 0)
                    self.encode_tokens(utils.split_pattern.split(r[0]), out=candidate_relations_words_matrix[i, j, k])
                    if len(entities_by_rel.get(r[1], [])) > 0:
                        e = entities_by_rel[r[1]][0]
                        candidate_entities_matrix[i, j, k] = self._entity2idx.get(e[1], 0)
                candidate_matrix[i, j] = self._entity2idx.get(l['kbID'], 0)
                self.encode_mention(l['label'], out=candidate_label_matrix[i, j])
                l_norm = self._normalize_candidate_features(l)
                for f, feature_name in enumerate(self._p['features_linking']):
                    candidate_features[i, j, f] = l_norm.get(feature_name, 1.0) + utils.epsilon
        return (sentences_matrix, mention_matrix,
                candidate_relations_matrix, candidate_relations_words_matrix,
                candidate_entities_matrix, candidate_matrix, candidate_label_matrix, candidate_features)

    def _get_torch_net(self):
        return VectorNet(self._p, self._embedding_matrix, self._entities_embedding_matrix, self._rels_embedding_matrix)
