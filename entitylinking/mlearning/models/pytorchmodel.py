import abc
import json
import time
from collections import Counter

import datetime
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn as nn
from torch.nn import functional as F

from colorama import Fore

from entitylinking import utils
from entitylinking.base_objects import Loggable
from entitylinking.core import candidate_retrieval
from entitylinking.evaluation import measures


def init_weights(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        m.weight.data.normal_(mean=0, std=np.sqrt(1/n))
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv1d):
        n = m.in_channels
        m.weight.data.normal_(mean=0, std=np.sqrt(1/n))
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Embedding):
        m.weight.data.normal_(mean=0, std=0.02)


class CustomCombinedLoss(nn.Module):

    def __init__(self, weight):
        super(CustomCombinedLoss, self).__init__()
        self._weight = weight
        self._criterion_choice = nn.MultiMarginLoss(size_average=False, margin=0.5)

    def forward(self, input, target):
        positive_prob, predictions = input
        positive_target = (target != 0).float()
        target_predictions = target - 1

        positive_targets_weights = torch.ones_like(positive_target).float()
        for i in range(positive_targets_weights.size(0)):
            if positive_target[i].data[0] < 1:
                positive_targets_weights[i] = self._weight
        loss = F.binary_cross_entropy(positive_prob, positive_target, weight=positive_targets_weights,
                                      size_average=False)
        for i in range(target.size(0)):
            if target_predictions[i].data[0] > -1:
                loss += self._criterion_choice(predictions[i], target_predictions[i])
        return loss


class PytorchModel(Loggable, metaclass=abc.ABCMeta):

    def __init__(self, parameters, *args, **kwargs):
        super(PytorchModel, self).__init__(*args, **kwargs)
        self._p = {**parameters}
        self._save_model_to = self._p['models.save.path']
        self._model = None
        self._file_extension = "torchweights"
        self._model_number = 0
        now = datetime.datetime.now()
        self._model_file_name = f"{self.__class__.__name__}_{now.date()}_{now.microsecond}.{self._file_extension}"
        self.logger.info("Assigned model file name: {}".format(self._model_file_name))
        self._optimizer = None
        self._criterion = None
        self._log_interval = 5
        self._batch_size = self._p.get("batch.size", 128)

        self._embedding_matrix = None
        self._word2idx = {}
        self._p['word.emb.size'] = 50

        self._criterion = None

    def prepare_model(self, embedding_matrix=None, element2idx=None):
        self.logger.info("Model parameters: " + str(self._p))
        self._model = self._get_torch_net()
        self._model.apply(init_weights)
        parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        self._optimizer = torch.optim.Adam(parameters, weight_decay=0)
        if torch.cuda.is_available():
            self._model.cuda()

    @abc.abstractmethod
    def _get_torch_net(self):
        """
        Initialize the structure of a Torch model and return an instance of a model to train. It may store other models
        as instance variables.
        :return: a Torch nn.Module to be trained
        """
        pass

    def _torchify_data(self, evaluation, *args):
        matrices = list(args)
        for i, array in enumerate(args):
            matrices[i] = torch.from_numpy(array)
            if torch.cuda.is_available():
                matrices[i] = matrices[i].cuda()
            matrices[i] = Variable(matrices[i], volatile=evaluation)
        return tuple(matrices)

    def index_to_one_hop(self, integer_matrix):
        matrix_one_hot = np.zeros((integer_matrix.shape[0], integer_matrix.shape[1], self._p['vocab.size']), dtype='uint8')
        first_index = np.arange(integer_matrix.shape[0]).reshape((integer_matrix.shape[0], 1, 1))
        second_index = np.arange(integer_matrix.shape[1]).reshape((1, integer_matrix.shape[1], 1))
        matrix_one_hot[first_index, second_index, integer_matrix] = 1
        return matrix_one_hot

    @abc.abstractmethod
    def encode_batch(self, data_without_targets, verbose=False):
        pass

    def scores_for_instance(self, data_encoded):
        self._model.eval()
        entities_num = len(data_encoded[0])
        choices = data_encoded[-1].shape[1]
        global_predictions = np.zeros((entities_num, choices + 1))
        for batch_i, pos in enumerate(range(0, entities_num, self._batch_size)):
            batch_encoded = self._get_batch(pos, *data_encoded)
            if all(len(m) > 0 for m in batch_encoded):
                batch_encoded = self._torchify_data(True, *batch_encoded)
                positive_prob, predictions = self._model(*batch_encoded)
                if torch.cuda.is_available():
                    predictions = predictions.cpu()
                    positive_prob = positive_prob.cpu()
                predictions = predictions.data.numpy()
                positive_prob = positive_prob.data.numpy()
                global_predictions[pos:pos+batch_encoded[0].size(0), 0] = 1 - positive_prob
                global_predictions[pos:pos+batch_encoded[0].size(0), 1:] = predictions
        return global_predictions

    def _save_model(self):
        with open(self._save_model_to + self._model_file_name.replace(self._file_extension, "param"), 'w') as out:
            json.dump(self._p, out, indent=2)
        state_dict = self._model.state_dict()
        if torch.cuda.is_available():
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
        torch.save(state_dict, self._save_model_to + self._model_file_name)

    def load_from_file(self, path_to_model):
        """
        Load a model from file.

        :param path_to_model: path to the model file.
        """
        load_from_file = path_to_model.replace(self._file_extension, "param")
        self.logger.info("Loading model from file, parameters file: {}".format(load_from_file))
        with open(load_from_file) as f:
            self._p.update(json.load(f))
        self.logger.info("Model parameters: " + str(self._p))

        self._model = self._get_torch_net()
        parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        self._optimizer = torch.optim.Adam(parameters)
        self._model.load_state_dict(torch.load(path_to_model))
        if torch.cuda.is_available():
            self._model.cuda()
        self.logger.info("Loaded successfully.")

    def load_last_saved(self):
        self.load_from_file(self._save_model_to + self._model_file_name)

    def _model_check(self, loss_history):
        loss_history = self._preprocess_history(loss_history)
        if 0 < len(loss_history) < 2 or loss_history[-1] < np.min(loss_history[:-1]):
            print(Fore.GREEN + " ↑", end="")
            if self._p.get("model.checkpoint", False):
                self._save_model()
                print(" Model saved", end="")
        else:
            print(Fore.RED + " ↓", end="")
        print(Fore.RESET, end="")

    def _early_stopping(self, loss_history):
        patience = self._p.get("early.stopping", 3)
        if len(loss_history) - 1 < patience:
            return False
        if self._p.get("evaluate.on.dataset.after", -1) > 0 \
                and len(loss_history) - patience < self._p.get("evaluate.on.dataset.after", -1):
            return False
        loss_history = loss_history[len(loss_history)-patience-1:]
        loss_history = self._preprocess_history(loss_history)
        better = any(loss_history[i] < loss_history[0] or loss_history[0] < 0.0 for i in range(1, patience+1))
        return not better

    def _get_batch(self, pos, *args):
        matrices = list(args)
        assert len(matrices) > 0
        master_m = matrices[0]
        assert all(len(m) == len(master_m) for m in matrices)
        batch_size = min(self._p.get("batch.size", 128), len(master_m) - pos)
        for i, m in enumerate(matrices):
            matrices[i] = m[pos:pos+batch_size]
        return tuple(matrices)

    def _fit_batch(self, batch_targets, *args):
        self._model.train()
        self._optimizer.zero_grad()
        positive_prob, predictions = self._model(*args)
        # negative_prob, predictions = F.softmax(predictions)
        loss = self._criterion((positive_prob, predictions), batch_targets)
        loss.backward()
        self._optimizer.step()
        cur_loss = loss.data[0]
        if torch.cuda.is_available():
            predictions = predictions.cpu()
            batch_targets = batch_targets.cpu()
            positive_prob = positive_prob.cpu()
        predictions = predictions.data.numpy()
        positive_prob = positive_prob.data.numpy()
        batch_targets = batch_targets.data.numpy()
        cur_acc, batch_predicted_targets = self._compute_eval_metrics(batch_targets, (positive_prob, predictions))
        return cur_loss, cur_acc, batch_predicted_targets

    def _compute_eval_metrics(self, batch_targets, predictions):
        positive_prob, candidate_predictions = predictions
        predicted_targets = np.argmax(candidate_predictions, axis=-1) + 1
        predicted_targets[positive_prob < 0.5] = 0
        cur_acc = np.sum(predicted_targets == batch_targets)
        cur_acc /= len(batch_targets)
        return cur_acc, predicted_targets

    def _preprocess_history(self, loss_history):
        loss_history = [l[self._p.get("monitor", "loss")] for l in loss_history]
        if self._p.get("monitor", "loss").split("_")[1] in {"acc", "f1"}:
            loss_history = [1.0 - l for l in loss_history]
        return loss_history

    def evaluate_batchwise(self, targets, *args):
        self._model.eval()
        running_loss = 0.0
        accuracy = 0.0
        predictions = []
        eval_set_size = len(targets)
        for batch_i, dataset_position in enumerate(range(0, eval_set_size, self._batch_size)):
            batch_input = self._get_batch(dataset_position, targets, *args)
            batch_input = self._torchify_data(True, *batch_input)
            batch_targets = batch_input[0]
            batch_positive_prob, batch_predictions = self._model(*batch_input[1:])
            # batch_predictions = F.softmax(batch_predictions)
            running_loss += self._criterion((batch_positive_prob, batch_predictions), batch_targets).data[0] * len(batch_targets)

            if torch.cuda.is_available():
                batch_predictions = batch_predictions.cpu()
                batch_positive_prob = batch_positive_prob.cpu()
                batch_targets = batch_targets.cpu()
            batch_predictions = batch_predictions.data.numpy()
            batch_positive_prob = batch_positive_prob.data.numpy()
            batch_targets = batch_targets.data.numpy()
            cur_accuracy, batch_predicted_targets = self._compute_eval_metrics(batch_targets, (batch_positive_prob, batch_predictions))
            cur_accuracy *= len(batch_targets)
            accuracy += cur_accuracy
            predictions.append(batch_predicted_targets)

        predictions = np.concatenate(predictions, axis=0)
        running_loss /= eval_set_size
        accuracy /= eval_set_size
        return running_loss, accuracy, predictions

    def train(self, train, dev=None, num_epochs=-1, eval_on_dataset=None):
        assert self._model is not None
        self.logger.info('Training process started.')

        evaluate_on_val = dev is not None and len(dev[-1]) > 0
        if evaluate_on_val:
            self.logger.debug("Start training with a validation sample.")
            val_encoded = dev[0]
            self.logger.debug("Encoded evaluation sentence matrix, shapes: {}".format(
                [el.shape for el in val_encoded]))
            targets_val = np.asarray(dev[-1])
            self.logger.info('Validating on {} samples.'.format(len(targets_val)))
        else:
            self.logger.debug("Start training without a validation sample.")

        targets = np.asarray(train[-1])
        training_set_size = len(targets)
        train_encoded = train[0]
        assert len(train_encoded[0]) == training_set_size
        self.logger.info('Training on {} samples.'.format(training_set_size))
        val_predictions = []

        num_batches = len(targets) // self._batch_size
        num_epochs = self._p.get("epochs", 200) if num_epochs == -1 else num_epochs
        assert num_epochs > 0
        loss_history = []
        epoch = 0
        while not self._early_stopping(loss_history) and epoch < num_epochs:
            # Fit batches
            running_loss = 0.0
            running_accuracy = 0.0
            epoch_time = time.time()
            for batch_i, dataset_position in enumerate(range(0, training_set_size, self._batch_size)):
                print('| E {:3d}/{:3d} | B {:5d}/{:5d} '.format(epoch, num_epochs, batch_i, num_batches), end='')
                batch_time = time.time()
                batch_input = self._get_batch(dataset_position, targets, *train_encoded)
                batch_input = self._torchify_data(False, *batch_input)
                batch_targets = batch_input[0]
                current_results = self._fit_batch(batch_targets, *batch_input[1:])
                cur_loss, cur_acc = current_results[0], current_results[1]
                running_loss += cur_loss * len(batch_targets)
                running_accuracy += cur_acc * len(batch_targets)
                elapsed = time.time() - batch_time
                print('| s/b {:5.2f} | l {:4.4f} | a {:4.4f} |'.format(elapsed, cur_loss, cur_acc), end='\r')
            running_loss /= training_set_size
            running_accuracy /= training_set_size
            elapsed = time.time() - epoch_time
            print('| E {:3d}/{:3d} | B {:5d}/{:5d} | s/e {:5.2f} | L {:4.4f} | A {:4.4f} |'
                  .format(epoch, num_epochs, num_batches, num_batches, elapsed, running_loss, running_accuracy), end='')
            # Evaluate on validation set
            loss_dict = {'t_loss': running_loss, 't_acc': running_accuracy}
            if evaluate_on_val:
                val_loss, val_accuracy, val_predictions = self.evaluate_batchwise(targets_val, *val_encoded)
                v_prec, v_rec, v_f1 = measures.prec_rec_f1(val_predictions, targets_val, empty_guessed=0)
                positive_target = (targets_val != 0)
                positive_prediction = (val_predictions != 0)
                v_acc_neg = np.sum(positive_target == positive_prediction) / len(positive_target)
                print(' vL {:4.4f} | vA {:2.4f}, vN {:2.4f} | vF1 {:2.3f},{:2.3f},{:2.3f} |'.format(
                    val_loss, val_accuracy, v_acc_neg, v_prec, v_rec, v_f1), end='')
                loss_dict.update({"v_loss": val_loss, "v_acc": val_accuracy, "v_f1": v_f1})
                if eval_on_dataset is not None:
                    if epoch > self._p.get("evaluate.on.dataset.after", -1):
                        dataset_results = eval_on_dataset()
                    else:
                        dataset_results = (-1, -1, -1)
                    print(' dF1 {:2.3f},{:2.3f},{:2.3f} |'.format(*dataset_results), end='')
                    loss_dict['d_f1'] = dataset_results[2]
            loss_history.append(loss_dict)
            epoch += 1
            self._model_check(loss_history)
            print("")
        if epoch != num_epochs:
            print("Early stopping")
        self.logger.info("Model training is finished.")
        if len(val_predictions) > 0:
            prediction_freq = Counter(val_predictions)
            self.logger.info("Validation, predicted classes: {}".format(prediction_freq.most_common(10)))
        if self._p.get("model.checkpoint", False):
            self.logger.info("Best model saved to: {}".format(self._save_model_to))
        return loss_history[-1]


class ELModel(PytorchModel, metaclass=abc.ABCMeta):

    def __init__(self, parameters, *args, **kwargs):
        super(ELModel, self).__init__(parameters, *args, **kwargs)

        self._p["negative.class.weight"] = self._p.get("negative.class.weight", 0.01)
        self._criterion = CustomCombinedLoss(self._p["negative.class.weight"])  # type: nn.Module
        scale_type = self._p.get("negative.weight.scale.type", "no")
        weight_epoch = self._p.get("negative.weight.epoch", 5)
        if scale_type == "smooth":
            torch_weight = sigmoid(-5) * self._p["negative.class.weight"]
            self._criterion = CustomCombinedLoss(weight=torch_weight)
        elif scale_type == "linear":
            torch_weight = 1/weight_epoch * self._p["negative.class.weight"]
            self._criterion = CustomCombinedLoss(weight=torch_weight)
        elif scale_type == "step":
            self._criterion = CustomCombinedLoss(weight=self._p["negative.class.weight"] * 0.01)
        else:
            self._criterion = CustomCombinedLoss(weight=self._p["negative.class.weight"])

    def _model_check(self, loss_history):
        super(ELModel, self)._model_check(loss_history)
        weight_epoch = self._p.get("negative.weight.epoch", 5)
        scale_type = self._p.get("negative.weight.scale.type", "no")
        if scale_type == "smooth":
            if 1 < len(loss_history) < weight_epoch:
                torch_weight = sigmoid(len(loss_history)/(weight_epoch*0.1) - 5) * self._p["negative.class.weight"]
                self._criterion = CustomCombinedLoss(weight=torch_weight)
        elif scale_type == "linear":
            if 1 < len(loss_history) < weight_epoch:
                torch_weight = len(loss_history)/weight_epoch * self._p["negative.class.weight"]
                self._criterion = CustomCombinedLoss(weight=torch_weight)
        elif scale_type == "step":
            if len(loss_history) == weight_epoch:
                self._criterion = CustomCombinedLoss(weight=self._p["negative.class.weight"])

    def prepare_model(self, embedding_matrix=None, element2idx=None):
        if embedding_matrix is None and element2idx is None:
            self.logger.debug("Loading embeddings")
            self._embedding_matrix, self._word2idx = utils.load_word_embeddings(self._p['word.embeddings'])
        else:
            self._embedding_matrix, self._word2idx = embedding_matrix, element2idx
        self.logger.info("Loaded embeddings: {}".format(self._embedding_matrix.shape))

        self._p['word.emb.size'] = self._embedding_matrix.shape[1]
        self._p['word.vocab.size'] = len(self._word2idx)
        super(ELModel, self).prepare_model()

    def load_from_file(self, path_to_model):
        """
        Load a model from file.

        :param path_to_model: path to the model file.
        """
        super(ELModel, self).load_from_file(path_to_model)
        if self._embedding_matrix is None:
            self.logger.debug("Loading embeddings")
            self._embedding_matrix, self._word2idx = utils.load_word_embeddings(self._p['word.embeddings'])
        self.logger.debug("Loaded embeddings: {}".format(self._embedding_matrix.shape))
        self._p['word.emb.size'] = self._embedding_matrix.shape[1]
        self._p['word.vocab.size'] = len(self._word2idx)

    def _normalize_candidate_features(self, l):
        l = dict(**l)
        l['freq'] = l.get('freq', 0) / candidate_retrieval.max_entity_freq
        l['id_rank'] = _normalize_dict_feature(l, 'id_rank', np.log(4 * 10 ** 7).item())
        l['lev_main_label'] = _normalize_dict_feature(l, 'lev_main_label', 30)
        l['lev_matchedlabel'] = _normalize_dict_feature(l, 'lev_matchedlabel', 30)
        l['lev_sentence'] = _normalize_dict_feature(l, 'lev_sentence', 30)
        l['match_diff'] = _normalize_dict_feature(l, 'match_diff', 10)
        l['mention_tokens_len'] = min((l['mention_tokens_len'] / self._p['max.ngram.len']), 1.0)
        l['singature_overlap_score'] = _normalize_dict_feature(l, 'singature_overlap_score', 10)
        l['num_related_entities'] = _normalize_dict_feature(l, 'num_related_entities', 1000)
        l['num_related_relations'] = _normalize_dict_feature(l, 'num_related_relations', 1000)
        return l


def _normalize_dict_feature(d, f_name, max_value):
    return min(d[f_name] / max_value, 1.0) if f_name in d else 0.0


def _normalize_entity_features(e):
    e = dict(**e)
    e['fragment_tokens_len'] = min(len(e.get("tokens", [])) / 20, 1.0)
    return e


def sigmoid(x):
    return 1/(1 + np.exp(-x))
