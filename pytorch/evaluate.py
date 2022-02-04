from sklearn import metrics

from pytorch_utils import forward
from torch.nn.functional import one_hot
from torch import from_numpy
import copy
import numpy as np
from losses import get_loss_func

class Evaluator(object):
    def __init__(self, model, loss_type='clip_ce'):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        self.loss_type = loss_type
        self.loss_func = get_loss_func(loss_type)

    def evaluate(self, data_loader, use_audeep=False, return_embeddings=False, do_binary_task= False, return_tagging_predictions_and_targets=False):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict,
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model,
            generator=data_loader,
            use_audeep=use_audeep,
            return_target=True)


        audio_names = output_dict['audio_name']
        embeddings = output_dict["embedding"]
        clipwise_output = output_dict['clipwise_output']  # (audios_num, classes_num)
        clipwise_output_tensor = output_dict['clipwise_output_tensor']  # (audios_num, classes_num)
        target = output_dict['target']  # (audios_num, classes_num)
        target_copy = copy.deepcopy(target)

        if self.loss_type == 'clip_ce':
            if not do_binary_task:
                onehot_target = one_hot(from_numpy(target_copy)).squeeze().numpy()
                # print(target.shape, onehot_target.shape)
                uar = metrics.recall_score(target, np.argmax(clipwise_output, axis=1), average='macro')

            else:
                onehot_target = target_copy
                uar = -1.

            average_precision = metrics.average_precision_score(
                onehot_target, clipwise_output, average=None)

            auc = metrics.roc_auc_score(onehot_target, clipwise_output, average=None)

        elif self.loss_type == 'clip_bce' or 'clip_nll':
            """in this case, target is one hot"""

            average_precision = metrics.average_precision_score(
                target_copy, clipwise_output, average=None)

            if not do_binary_task:
                uar = metrics.recall_score(np.argmax(target_copy, axis=-1), np.argmax(clipwise_output, axis=1), average='macro')
            else:
                uar = -1.0

            auc = metrics.roc_auc_score(target_copy, clipwise_output, average=None)

        output_dict_tensor ={'clipwise_output': clipwise_output_tensor}

        loss = self.loss_func(output_dict_tensor, output_dict)

        statistics = {'average_precision': average_precision, 'auc': auc, 'uar': uar, 'loss': loss}

        # print('ten_first_labels', target[:10])
        # print('ten_first_probs', clipwise_output[:10])

        if return_tagging_predictions_and_targets and return_embeddings:
            return statistics, clipwise_output, target, audio_names, embeddings
        elif return_tagging_predictions_and_targets:
            return statistics, clipwise_output, target, audio_names
        else:
            return statistics

    # def long_to_onehot(self, t):
