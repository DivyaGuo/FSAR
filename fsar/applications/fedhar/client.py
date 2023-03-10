import logging
import os
import time

import copy
import wandb

import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from easyfl.client.base import BaseClient
from easyfl.distributed.distributed import CPU
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.tracking import metric
from model import get_classifier
from model import get_unshareC

logger = logging.getLogger(__name__)


def KD(input_p, input_q, T=1):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    p = F.softmax(input_p / T, dim=1)
    q = F.log_softmax(input_q / T, dim=1)
    result = kl_loss(q, p)
    return result


class FedHARClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, sleep_time=0, topk=1):
        super(FedHARClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time)
        self.topk = topk

        self.classifier = get_classifier(len(self.train_data.classes[cid])).to(device)
        self.unshareC = get_unshareC()

        # logs
        self.global_step = {}
        self.global_step[cid] = 0
        self.epoch = {}
        self.epoch[cid] = 0

        # wandb
        file_path = os.path.join(os.getcwd(), "experiments", "exp" + time.strftime("_%m_%d_%H_%M"))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.wandblog = wandb.init(config=conf,
                                   project="FL-HAR",
                                   entity="FSAR",
                                   dir=file_path,
                                   name='FSAR_experiment000')
        # loss_ce
        self.args_lambda_ce = 1

        # loss_reg
        self.args_lambda_reg = 1
        self.mu = 0.01

        # loss_kd, loss_ce
        self.args_temp = 1
        self.args_lambda_branch_ce = 0.5
        self.args_lambda_branch_kl = 0.5

    def train(self, conf, device=CPU):

        fixed_model = copy.deepcopy(self.model).to(device)
        for param_t in fixed_model.parameters():
            param_t.requires_grad = False

        self.model.classifier.classifier = self.classifier.to(device)
        self.model.model.unshareC = self.unshareC.to(device)

        global_model = copy.deepcopy(self.model).to(device)
        for par in global_model.parameters():
            par.requires_grad = False

        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.loss_fn = loss_fn
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        epoch_loss = []
        result_frag = []
        label_frag = []

        for i in range(conf.local_epoch):
            batch_loss = []
            cnt = 0
            for data, label in self.train_loader:
                # get data
                data = data.float().to(device)
                label = label.long().to(device)

                features_from_globel_model  = global_model(data, mode='global')
                NM_for_train_model = features_from_globel_model[-1]

                # loss_ce
                main_branch_logits = self.model(data, mode='train', NM_for_train_model=NM_for_train_model)
                loss_ce = loss_fn(main_branch_logits, label)

                # loss_kd, loss_ce
                num_level_to_align = 1
                ce_branch = []
                kl_branch = []
                for level in range(num_level_to_align):
                    start_level = level
                    branch_logit = self.model(features_from_globel_model[level], mode='train',
                                              start_level=start_level,
                                              NM_for_train_model=NM_for_train_model)
                    this_ce = loss_fn(branch_logit, label)
                    this_kl = KD(branch_logit, main_branch_logits, self.args_temp)
                    ce_branch.append(this_ce)
                    kl_branch.append(this_kl)
                if num_level_to_align == 0: num_level_to_align += 1
                loss_branch_ce = sum(ce_branch) / num_level_to_align
                loss_branch_kl = sum(kl_branch) / num_level_to_align

                # loss_reg
                loss_reg = 0
                fixed_params = {n: p for n, p in fixed_model.named_parameters()}
                for n, p in self.model.named_parameters():
                    if 'unshare' in n: continue
                    if n in fixed_params:
                        loss_reg += ((p - fixed_params[n].detach()) ** 2).sum()

                loss = self.args_lambda_ce * loss_ce + \
                       self.args_lambda_reg * 0.5 * self.mu * loss_reg + \
                       self.args_lambda_branch_ce * loss_branch_ce + \
                       self.args_lambda_branch_kl * loss_branch_kl

                cnt += 1
                if cnt % 100 == 0:
                    logger.info("ce_loss:{}  reg_loss:{}  branch_ce_loss:{}  branch_kl_loss:{}".format(
                        self.args_lambda_ce * loss_ce,
                        self.args_lambda_reg * 0.5 * self.mu * loss_reg,
                        self.args_lambda_branch_ce * loss_branch_ce,
                        self.args_lambda_branch_ce * loss_branch_ce))


                # compute accuracy
                result_frag.append(main_branch_logits.data.cpu().numpy())
                label_frag.append(label.data.cpu().numpy())

                # backword
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                global_model.classifier.classifier.load_state_dict(self.model.classifier.classifier.state_dict())
                for par in global_model.parameters():
                    par.requires_grad = False

                break

            self.result = np.concatenate(result_frag)
            self.label = np.concatenate(label_frag)
            accuracy = self.show_topk(self.topk)

            scheduler.step()
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(float(current_epoch_loss))
            logger.info("Client {}, local epoch: {}, loss: {:.2f}, accuracy: {:.2f}%".format(self.cid, i, current_epoch_loss, accuracy * 100))

            self.global_step[self.cid] = self.global_step[self.cid] + len(self.train_loader)
            self.epoch[self.cid] = self.epoch[self.cid] + 1

            self.wandblog.log({self.cid + '_train_loss': current_epoch_loss,
                               self.cid + '_train_accuracy': accuracy * 100,
                               self.cid + '_train_global_step': self.global_step[self.cid],
                               self.cid + '_train_epoch': self.epoch[self.cid], })

        # logs
        self.current_round_time = time.time() - start_time
        self.track(metric.TRAIN_TIME, self.current_round_time)
        self.track(metric.TRAIN_LOSS, epoch_loss)

        self.classifier = self.model.classifier.classifier
        self.model.classifier.classifier = nn.Sequential()

        self.unshareC = self.model.model.unshareC
        self.model.model.unshareC = nn.Parameter()
        self.model.to(device)

        return self.model.state_dict()

    def test(self, conf, device=CPU):

        self.model.classifier.classifier = self.classifier.to(device)
        self.model.model.unshareC = self.unshareC.to(device)

        self.model = self.model.eval()
        self.model = self.model.to(device)

        loss_value = []
        result_frag = []
        label_frag = []

        if self.test_loader is None:
            self.test_loader = self.test_data.loader(conf.test_batch_size, self.cid, shuffle=False, seed=conf.seed)
        for data, label in self.test_loader:
            # get data
            data = data.float().to(device)
            label = label.long().to(device)

            # inference
            with torch.no_grad():
                out = self.model(data, mode='test')
            result_frag.append(out.data.cpu().numpy())

            # get loss
            loss = self.loss_fn(out, label)
            loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())

            break

        self.model.classifier.classifier = nn.Sequential()
        self.model.model.unshareC = nn.Parameter()
        self.model.to(device)

        # get mean loss
        mean_loss = np.mean(loss_value)

        # show top-k accuracy
        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        accuracy = self.show_topk(self.topk)

        # logs
        logger.info("Dataset {} Loss:{:.2f} Top{}:{:.2f}%".format(
            self.cid, mean_loss, self.topk, 100 * accuracy))

        self.wandblog.log({self.cid + '_test_loss': mean_loss,
                           self.cid + '_test_accuracy': 100 * accuracy, })

        # communication
        self._upload_holder = server_pb.UploadContent(
            data=codec.marshal(server_pb.Performance(accuracy=accuracy, loss=mean_loss)),  # loss not applicable
            type=common_pb.DATA_TYPE_PERFORMANCE,
            data_size=1,
        )

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        return accuracy

    def load_optimizer(self, conf):
        ignored_params = list(map(id, self.model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * conf.optimizer.lr},
            {'params': self.model.classifier.parameters(), 'lr': conf.optimizer.lr}
        ], weight_decay=5e-4, momentum=conf.optimizer.momentum, nesterov=True)
        return optimizer_ft
