import torch.multiprocessing

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual, visual_anomaly, adjustment, find_segment_lengths, find_segments, \
    visual_anomaly_segment, visual_anomaly_segment_MS, visual_anomaly_segment_Multi
from utils.tsne import visualization,visualization_PCA
from utils.anomaly_detection_metrics import adjbestf1,f1_score
from utils.ensemble import CRPSmetric
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import collections
from collections import Counter
import csv
from tqdm import *
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_AR(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_AR, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        if self.args.freeze_decoder:
            for name, param in model.decoder.named_parameters():
                param.requires_grad = False
            for name, param in model.backbone.decoder.named_parameters():
                param.requires_grad = False
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ims:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                    # input and output are completely aligned
                    if self.args.use_mask:
                        # masked reconstruction task
                        # random mask
                        B, T, N = batch_x.shape
                        assert T % self.args.patch_len == 0
                        mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                        mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                        mask[mask <= self.args.mask_rate] = 0  # masked
                        mask[mask > self.args.mask_rate] = 1  # remained
                        mask = mask.view(mask.size(0), -1, mask.size(-1))
                        mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                        inp = batch_x.masked_fill(mask == 0, 0)

                        outputs = self.model(inp[:, self.args.patch_len:, :], None, None, None, mask)
                    else:
                        outputs = self.model(batch_x[:, self.args.patch_len:, :], None, None, None)

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def finetune(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if type(batch_x) is list:
                    batch_x = batch_x[0]
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ensemble_forecast:
                    output_ensemble_list = []
                    for token_idx in range(rec_token_count):
                        outputs_ensemble = self.model(batch_x[:, token_idx*self.args.patch_len:-self.args.patch_len, :], None, None, None)[:,-self.args.patch_len:, :]
                        output_ensemble_list.append(outputs_ensemble)
                    outputs = torch.cat(output_ensemble_list, dim=2)
                    if self.args.ensemble_type == 'mean':
                        # Merge ensemble with MEAN
                        outputs = torch.mean(outputs, dim=2, keepdim=True)
                    batch_x = batch_x[:, -self.args.patch_len:, :]
                    
                    """
                    ?: numpy version CRPS metric does not support back-propogation
                    !: the loss used in training and testing might be different, such as classification trained with CE and tested with Accuracy
                    """
                    # loss = torch.mean(CRPSmetric(outputs, batch_x))
                    loss = criterion(outputs, batch_x) # mean of ensemble by default
                else:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                    loss = criterion(outputs, batch_x)
                
                train_loss.append(loss.item())

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def find_border(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border1_str = parts[-2]
        border2_str = parts[-1]
        if '.' in border2_str:
            border2_str = border2_str[:border2_str.find('.')]

        try:
            border1 = int(border1_str)
            border2 = int(border2_str)
            return border1, border2
        except ValueError:
            return None

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        score_list = []
        if self.args.is_finetuning:
            status='_adaption'
        else:
            status='_zero-shot'
        status += '_AR'
        folder_path = './test_results/' + self.args.data + status +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_path = os.path.join(folder_path, setting)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        time_now = time.time()

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        if self.args.data == 'UCRA':
            border_start = self.find_border_number(self.args.data_path)
            border1, border2 = self.find_border(self.args.data_path)

        token_count = 0
        rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len

        input_list = []
        output_list = []
        test_labels = []
        embedding_list = []
        feature_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(test_loader),total=len(test_loader),leave = True):
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ensemble_forecast:
                    output_ensemble_list = []
                    for token_idx in range(rec_token_count):
                        outputs_ensemble = self.model(batch_x[:, token_idx*self.args.patch_len:-self.args.patch_len, :], None, None, None)[:,-self.args.patch_len:, :]
                        output_ensemble_list.append(outputs_ensemble)
                    # outputs = torch.cat(output_ensemble_list, dim=2)
                    outputs = torch.cat(output_ensemble_list, dim=0)

                    if self.args.ensemble_type == 'mean':
                        # Merge ensemble with MEAN in testing
                        outputs = torch.mean(outputs, dim=0, keepdim=True)
                        
                    output_list.append(outputs[0, :, :].detach().cpu().numpy())
                    batch_x = batch_x[:, -self.args.patch_len:, :]
                    batch_y = batch_y[:, -self.args.patch_len:]
                    input_list.append(batch_x[0, :, :].detach().cpu().numpy())
                    test_labels.append(batch_y.reshape(-1).detach().cpu().numpy())
                    if self.args.ensemble_type == 'crps':
                        # calculate anomaly score with CRPS in testing
                        score = CRPSmetric(outputs,batch_x)
                        """ Use mean for visualization, this might be updated to better visualization of mu & sigma """
                        outputs = torch.mean(outputs, dim=2,keepdim=True)
                    else:
                        # calculate anomaly score with MSE in testing
                        score = self.anomaly_criterion(outputs, batch_x)
                    score_list.append(score[0,:,:].detach().cpu().numpy())


                    # outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)[:,-self.args.patch_len:, :]
                else:
                    #Hint! Shape matter: output (ensemble_num, seq_len, var_num)
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)[:, -self.args.patch_len:, :]
                    # output_list.append(outputs[0, :, -1].detach().cpu().numpy())
                    output_list.append(outputs[0, :, :].detach().cpu().numpy())
                    batch_x = batch_x[:, -self.args.patch_len:, :]
                    batch_y = batch_y[:, -self.args.patch_len:]
                    # input_list.append(batch_x[0, :, -1].detach().cpu().numpy())
                    input_list.append(batch_x[0, :, :].detach().cpu().numpy())
                    test_labels.append(batch_y.reshape(-1).detach().cpu().numpy())

                    score = self.anomaly_criterion(outputs, batch_x)
                    score_list.append(score[0,:,:].detach().cpu().numpy())
                    
                # if i % 50 == 0:
                #     cost_time = time.time() - time_now
                #     print(
                #         "\titers: {0}, cost_time: {1:.0f} | memory: allocated {2:.0f}MB, reserved {3:.0f}MB, cached {4:.0f}MB "
                #         .format(i,  cost_time,
                #                 torch.cuda.memory_allocated() / 1024 / 1024,
                #                 torch.cuda.memory_reserved() / 1024 / 1024,
                #                 torch.cuda.memory_cached() / 1024 / 1024))
                #     time_now = time.time()
        #* Evaluate metrics
        test_labels = np.concatenate(test_labels, axis=0).flatten()
        # test_labels = np.repeat(np.expand_dims(test_labels, 1), score.shape[-1], axis=1) # expand for multi-variate datasets
        # input = np.concatenate(input_list, axis=0).reshape(-1)
        # output = np.concatenate(output_list, axis=0).reshape(-1)
        # score_list = np.concatenate(score_list, axis=0).reshape(-1)
        input = np.concatenate(input_list, axis=0)
        output = np.concatenate(output_list, axis=0)
        score_list = np.concatenate(score_list, axis=0)

        
        # 输出adjustment best f1及best f1在最佳阈值下的原始结果
        best_pred_adj, best_pred = adjbestf1(test_labels.reshape(-1), score_list.mean(axis=-1), 100)
        # 计算没有adjustment的结果
        gt = test_labels.astype(int)
        accuracy = accuracy_score(gt, best_pred.astype(int))
        precision, recall, f_score, support = precision_recall_fscore_support(gt, best_pred.astype(int), average='binary')
        gt, adj_pred = adjustment(gt, best_pred_adj)
        adjaccuracy = accuracy_score(gt, adj_pred)
        adjprecision, adjrecall, adjf_score, adjsupport = precision_recall_fscore_support(gt, adj_pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        print("adjAccuracy : {:0.4f}, adjPrecision : {:0.4f}, adjRecall : {:0.4f}, adjF-score : {:0.4f} ".format(
            adjaccuracy, adjprecision, adjrecall, adjf_score))
        # accuracy = accuracy.astype(np.float32)
        # recall = recall.astype(np.float32)
        # f_score = f_score.astype(np.float32)
        # adjaccuracy = adjaccuracy.astype(np.float32)
        # adjprecision = adjprecision.astype(np.float32)
        # adjrecall = adjrecall.astype(np.float32)
        # adjf_score = adjf_score.astype(np.float32)
        # Write results to CSV file
        results = {
            "setting": [setting],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F-score': f_score,
            'adjAccuracy': adjaccuracy,
            'adjPrecision': adjprecision,
            'adjRecall': adjrecall,
            'adjF-score': adjf_score
        }
        # 将非迭代的值包装在列表中
        for key in results:
            if not isinstance(results[key], collections.abc.Iterable):
                results[key] = [results[key]]

        csv_file = data_path + '/' + 'results.csv'


        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))

        print("Results appended to", csv_file)
        

        #* visualization
        file_path_border = data_path + '/' + self.args.data_path[:self.args.data_path.find('.')] + '_AR_border.pdf'
        file_path = data_path + '/' + self.args.data_path[:self.args.data_path.find('.')] + '_AR_testset.pdf'
        visual_anomaly_segment_MS(input, output, best_pred, test_labels, file_path)
        
        if self.args.data == 'UCRA':
            input_border = input[border1 - border_start - self.args.patch_len*10 : border2 - border_start + self.args.patch_len*10]
            output_border = output[border1 - border_start - self.args.patch_len*10:border2 - border_start + self.args.patch_len*10]
            test_labels_border = test_labels[border1 - border_start - self.args.patch_len*10:border2 - border_start + self.args.patch_len*10]
            best_pred_border = best_pred[border1 - border_start - self.args.patch_len*10:border2 - border_start + self.args.patch_len*10]
            file_path_border = data_path + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_AR_border.pdf'
            visual_anomaly_segment(input_border, output_border, best_pred_border, test_labels_border, file_path_border)

        return
