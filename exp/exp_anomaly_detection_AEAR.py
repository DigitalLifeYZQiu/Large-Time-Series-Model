import torch.multiprocessing

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual, visual_anomaly, adjustment
from utils.tsne import visualization,visualization_PCA
from utils.anomaly_detection_metrics import adjbestf1,f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import collections
from collections import Counter
import csv

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from models import TrmEncoder, Timer


class Exp_Anomaly_Detection_AEAR(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TrmEncoder': TrmEncoder,
            'Timer': Timer,
        }
        (self.model_AE, self.model_AR) = self._build_model()
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self._acquire_device()
            self.model_AE.to(self.device)
            self.model_AR.to(self.device)

    def _build_model(self):
        model_AE = self.model_dict[self.args.model].Model(self.args).float()
        model_AR = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model_AE = nn.DataParallel(model_AE, device_ids=self.args.device_ids)
            model_AR = nn.DataParallel(model_AR, device_ids=self.args.device_ids)
        # if self.args.freeze_decoder:
        #     for name, param in model.decoder.named_parameters():
        #         param.requires_grad = False
        #     for name, param in model.backbone.decoder.named_parameters():
        #         param.requires_grad = False
        return (model_AE, model_AR)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim_AE = optim.Adam(self.model_AE.parameters(), lr=self.args.learning_rate)
        model_optim_AR = optim.Adam(self.model_AR.parameters(), lr=self.args.learning_rate)
        return (model_optim_AE, model_optim_AR)

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss_AE = []
        total_loss_AR = []
        self.model_AE.eval()
        self.model_AR.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
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

                    outputs_AE = self.model_AE(inp[:, self.args.patch_len:, :], None, None, None, mask)
                else:
                    outputs_AE = self.model_AE(batch_x[:, self.args.patch_len:, :], None, None, None)
                batch_x_AE = batch_x[:, self.args.patch_len:, :]
                outputs_AR = self.model_AR(batch_x[:, :-self.args.patch_len, :], None, None, None)
                batch_x_AR = batch_x[:, self.args.patch_len:, :]

                pred_AE = outputs_AE.detach().cpu()
                true_AE = batch_x_AE.detach().cpu()
                pred_AR = outputs_AR.detach().cpu()
                true_AR = batch_x_AR.detach().cpu()

                loss_AE = criterion(pred_AE, true_AE)
                loss_AR = criterion(pred_AR, true_AR)
                total_loss_AE.append(loss_AE)
                total_loss_AR.append(loss_AR)

        total_loss_AE = np.average(total_loss_AE)
        total_loss_AR = np.average(total_loss_AR)
        self.model_AE.train()
        self.model_AR.train()
        return (total_loss_AE, total_loss_AR)

    def finetune(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        (model_optim_AE, model_optim_AR) = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_AE = []
            train_loss_AR = []

            self.model_AE.train()
            self.model_AR.train()
            epoch_time = time.time()
            # for i, batch_x in enumerate(train_loader):
            #     iter_count += 1
            #     model_optim_AE.zero_grad()
            #     model_optim_AR.zero_grad()
            #     batch_x = batch_x.float().to(self.device)
            #     batch_x_AE = batch_x.float().to(self.device)
            #     batch_x_AR = batch_x.float().to(self.device)
                
            #     # input and output are completely aligned
            #     if self.args.use_mask:
            #         # masked reconstruction task
            #         # random mask
            #         B, T, N = batch_x_AE.shape
            #         assert T % self.args.patch_len == 0
            #         mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
            #         mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
            #         mask[mask <= self.args.mask_rate] = 0  # masked
            #         mask[mask > self.args.mask_rate] = 1  # remained
            #         mask = mask.view(mask.size(0), -1, mask.size(-1))
            #         mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
            #         inp = batch_x_AE.masked_fill(mask == 0, 0)

            #         outputs_AE = self.model_AE(inp[:, self.args.patch_len:, :], None, None, None, mask)
            #     else:
            #         outputs_AE = self.model_AE(batch_x_AE[:, self.args.patch_len:, :], None, None, None)

            #     batch_x_AE = batch_x_AE[:, self.args.patch_len:, :]
            #     outputs_AR = self.model_AR(batch_x_AR[:, :-self.args.patch_len, :], None, None, None)
            #     batch_x_AR = batch_x_AR[:, self.args.patch_len:, :]

            #     loss_AE = criterion(outputs_AE, batch_x_AE)
            #     loss_AR = criterion(outputs_AR, batch_x_AR)

            #     train_loss_AE.append(loss_AE.item())
            #     train_loss_AR.append(loss_AR.item())


            #     if i % 50 == 0:
            #         cost_time = time.time() - time_now
            #         print(
            #             "Auto-Encoding \t\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
            #             .format(i, epoch + 1, loss_AE.item(), cost_time,
            #                     torch.cuda.memory_allocated() / 1024 / 1024,
            #                     torch.cuda.memory_reserved() / 1024 / 1024,
            #                     torch.cuda.memory_cached() / 1024 / 1024))
            #         print(
            #             "Auto-Regression \titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
            #             .format(i, epoch + 1, loss_AR.item(), cost_time,
            #                     torch.cuda.memory_allocated() / 1024 / 1024,
            #                     torch.cuda.memory_reserved() / 1024 / 1024,
            #                     torch.cuda.memory_cached() / 1024 / 1024))
            #         time_now = time.time()

            #     loss_AE.backward()
            #     loss_AR.backward()
            #     model_optim_AE.step()
            #     model_optim_AR.step()
            #     torch.cuda.empty_cache()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim_AE.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_AE = batch_x.float().to(self.device)
                
                # input and output are completely aligned
                if self.args.use_mask:
                    # masked reconstruction task
                    # random mask
                    B, T, N = batch_x_AE.shape
                    assert T % self.args.patch_len == 0
                    mask = torch.rand((B, T // self.args.patch_len, N)).to(self.device)
                    mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                    mask[mask <= self.args.mask_rate] = 0  # masked
                    mask[mask > self.args.mask_rate] = 1  # remained
                    mask = mask.view(mask.size(0), -1, mask.size(-1))
                    mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                    inp = batch_x_AE.masked_fill(mask == 0, 0)

                    outputs_AE = self.model_AE(inp[:, self.args.patch_len:, :], None, None, None, mask)
                else:
                    outputs_AE = self.model_AE(batch_x_AE[:, self.args.patch_len:, :], None, None, None)

                batch_x_AE = batch_x_AE[:, self.args.patch_len:, :]

                loss_AE = criterion(outputs_AE, batch_x_AE)

                train_loss_AE.append(loss_AE.item())


                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "Auto-Encoding \t\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss_AE.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss_AE.backward()
                model_optim_AE.step()
                torch.cuda.empty_cache()

            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim_AR.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_AR = batch_x.float().to(self.device)
                
                outputs_AR = self.model_AR(batch_x_AR[:, :-self.args.patch_len, :], None, None, None)
                batch_x_AR = batch_x_AR[:, self.args.patch_len:, :]
                loss_AR = criterion(outputs_AR, batch_x_AR)
                train_loss_AR.append(loss_AR.item())


                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "Auto-Regression \titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss_AR.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()
                loss_AR.backward()
                model_optim_AR.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_AE = np.average(train_loss_AE)
            train_loss_AR = np.average(train_loss_AR)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Auto-Encoding Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss_AE, test_loss))
                print("Auto-Regression Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss_AR, test_loss))
            else:
                print("Auto-Encoding Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss_AE))
                print("Auto-Regression Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss_AR))

            adjust_learning_rate(model_optim_AE, epoch + 1, self.args)
            adjust_learning_rate(model_optim_AR, epoch + 1, self.args)

        return (self.model_AE, self.model_AR)

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
        score_list_AE = []
        score_list_AR = []
        if self.args.is_finetuning:
            status='_adaption'
        else:
            status='_zero-shot'
        folder_path_AE = './test_results/' + self.args.data + status +'_AE/'
        folder_path_AR = './test_results/' + self.args.data + status +'_AR/'
        if not os.path.exists(folder_path_AE) or not os.path.exists(folder_path_AR):
            os.makedirs(folder_path_AE)
            os.makedirs(folder_path_AR)

        self.model_AE.eval()
        self.model_AR.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        border_start = self.find_border_number(self.args.data_path)
        border1, border2 = self.find_border(self.args.data_path)

        token_count = 0
        # rec_token_count_AE = self.args.seq_len // self.args.patch_len
        rec_token_count_AE = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len
        rec_token_count_AR = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len
        

        input_list_AE = []
        input_list_AR = []
        output_list_AE = []
        output_list_AR = []
        test_labels_AE = []
        test_labels_AR = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs_AE = self.model_AE(batch_x[:, self.args.patch_len:-self.args.patch_len, :], None, None, None)
                batch_x_AE = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                batch_y_AE = batch_y[:, self.args.patch_len:-self.args.patch_len]
                outputs_AR = self.model_AR(batch_x[:, :-self.args.patch_len, :], None, None, None)[:, :-self.args.patch_len, :]
                batch_x_AR = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                batch_y_AR = batch_y[:, self.args.patch_len:-self.args.patch_len]

                input_list_AE.append(batch_x_AE[0, :, -1].detach().cpu().numpy())
                output_list_AE.append(outputs_AE[0, :, -1].detach().cpu().numpy())
                test_labels_AE.append(batch_y_AE.reshape(-1).detach().cpu().numpy())
                input_list_AR.append(batch_x_AR[0, :, -1].detach().cpu().numpy())
                output_list_AR.append(outputs_AR[0, :, -1].detach().cpu().numpy())
                test_labels_AR.append(batch_y_AR.reshape(-1).detach().cpu().numpy())
                # embedding_list.append(embeds.detach().cpu().numpy())
                # feature_list.append(features.detach().cpu().numpy())

                score_AE = self.anomaly_criterion(outputs_AE, batch_x_AE)
                score_AR = self.anomaly_criterion(outputs_AR, batch_x_AR)
                score_list_AE.append(score_AE.detach().cpu().numpy()) 
                score_list_AR.append(score_AR.detach().cpu().numpy()) 
                # for j in range(rec_token_count):
                #     # criterion
                #     token_start = j * self.args.patch_len
                #     token_end = token_start + self.args.patch_len
                #     score = torch.mean(self.anomaly_criterion(batch_x[:, token_start:token_end, :],
                #                                               outputs[:, token_start:token_end, :]), dim=-1)
                #     score = score.detach().cpu().numpy()
                #     score = np.mean(score)
                #     score_list.append((token_count, score))
                #     # embedding_list.append((token_count, embeds.detach().cpu().numpy()))
                #     token_count += 1
        
        #* Evaluate metrics
        test_labels_AE = np.concatenate(test_labels_AE, axis=0).flatten()
        test_labels_AR = np.concatenate(test_labels_AR, axis=0).flatten()
        input_AE = np.concatenate(input_list_AE, axis=0).reshape(-1)
        input_AR = np.concatenate(input_list_AR, axis=0).reshape(-1)
        output_AE = np.concatenate(output_list_AE, axis=0).reshape(-1)
        output_AR = np.concatenate(output_list_AR, axis=0).reshape(-1)
        score_list_AE = np.concatenate(score_list_AE, axis=0).reshape(-1)
        score_list_AR = np.concatenate(score_list_AR, axis=0).reshape(-1)
        
        # 输出adjustment best f1及best f1在最佳阈值下的原始结果
        best_pred_adj_AE, best_pred_AE = adjbestf1(test_labels_AE, score_list_AE, 100)
        best_pred_adj_AR, best_pred_AR = adjbestf1(test_labels_AR, score_list_AR, 100)
        # 计算没有adjustment的结果
        gt_AE = test_labels_AE.astype(int)
        gt_AR = test_labels_AR.astype(int)
        accuracy_AE = accuracy_score(gt_AE, best_pred_AE.astype(int))
        accuracy_AR = accuracy_score(gt_AR, best_pred_AR.astype(int))
        precision_AE, recall_AE, f_score_AE, support_AE = precision_recall_fscore_support(gt_AE, best_pred_AE.astype(int), average='binary')
        precision_AR, recall_AR, f_score_AR, support_AR = precision_recall_fscore_support(gt_AR, best_pred_AR.astype(int), average='binary')
        gt_AE, adj_pred_AE = adjustment(gt_AE, best_pred_adj_AE)
        gt_AR, adj_pred_AR = adjustment(gt_AR, best_pred_adj_AR)
        adjaccuracy_AE = accuracy_score(gt_AE, adj_pred_AE)
        adjaccuracy_AR = accuracy_score(gt_AR, adj_pred_AR)
        adjprecision_AE, adjrecall_AE, adjf_score_AE, adjsupport_AE = precision_recall_fscore_support(gt_AE, adj_pred_AE, average='binary')
        adjprecision_AR, adjrecall_AR, adjf_score_AR, adjsupport_AR = precision_recall_fscore_support(gt_AR, adj_pred_AR, average='binary')
        print("Auto-Encoding \tAccuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy_AE, precision_AE, recall_AE, f_score_AE))
        print("Auto-Encoding \tadjAccuracy : {:0.4f}, adjPrecision : {:0.4f}, adjRecall : {:0.4f}, adjF-score : {:0.4f} ".format(
            adjaccuracy_AE, adjprecision_AE, adjrecall_AE, adjf_score_AE))
        print("Auto-Regression \tAccuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy_AR, precision_AR, recall_AR, f_score_AR))
        print("Auto-Regression \tadjAccuracy : {:0.4f}, adjPrecision : {:0.4f}, adjRecall : {:0.4f}, adjF-score : {:0.4f} ".format(
            adjaccuracy_AR, adjprecision_AR, adjrecall_AR, adjf_score_AR))
        
        accuracy_AE = accuracy_AE.astype(np.float32)
        accuracy_AR = accuracy_AR.astype(np.float32)
        recall_AE = recall_AE.astype(np.float32)
        recall_AR = recall_AR.astype(np.float32)
        f_score_AE = f_score_AE.astype(np.float32)
        f_score_AR = f_score_AR.astype(np.float32)
        adjaccuracy_AE = adjaccuracy_AE.astype(np.float32)
        adjaccuracy_AR = adjaccuracy_AR.astype(np.float32)
        adjprecision_AE = adjprecision_AE.astype(np.float32)
        adjprecision_AR = adjprecision_AR.astype(np.float32)
        adjrecall_AE = adjrecall_AE.astype(np.float32)
        adjrecall_AR = adjrecall_AR.astype(np.float32)
        adjf_score_AE = adjf_score_AE.astype(np.float32)
        adjf_score_AR = adjf_score_AR.astype(np.float32)
        # Write results to CSV file
        results = {
            "setting": [setting],
            'Accuracy_AE': accuracy_AE,
            'Precision_AE': precision_AE,
            'Recall_AE': recall_AE,
            'F-score_AE': f_score_AE,
            'adjAccuracy_AE': adjaccuracy_AE,
            'adjPrecision_AE': adjprecision_AE,
            'adjRecall_AE': adjrecall_AE,
            'adjF-score_AE': adjf_score_AE,
            'Accuracy_AR': accuracy_AR,
            'Precision_AR': precision_AR,
            'Recall_AR': recall_AR,
            'F-score_AR': f_score_AR,
            'adjAccuracy_AR': adjaccuracy_AR,
            'adjPrecision_AR': adjprecision_AR,
            'adjRecall_AR': adjrecall_AR,
            'adjF-score_AR': adjf_score_AR,
        }
        # 将非迭代的值包装在列表中
        # for key in results:
        #     if not isinstance(results[key], collections.abc.Iterable):
        #         results[key] = [results[key]]

        # csv_file = 'results.csv'


        # with open(csv_file, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     if file.tell() == 0:
        #         writer.writerow(results.keys())
        #     writer.writerows(zip(*results.values()))

        # print("Results appended to", csv_file)

        #* visualization
        # half_patch_len = self.args.patch_len // 2
        # input_border = input[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        # output_border = output[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        border_start_AE = border_start
        border_start_AR = border_start
        input_border_AE = input_AE[border1 - border_start_AE - self.args.patch_len*5 : border2 - border_start_AE + self.args.patch_len*5]
        input_border_AR = input_AR[border1 - border_start_AR - self.args.patch_len*5 : border2 - border_start_AR + self.args.patch_len*5]
        output_border_AE = output_AE[border1 - border_start_AE - self.args.patch_len*5:border2 - border_start_AE + self.args.patch_len*5]
        output_border_AR = output_AR[border1 - border_start_AR - self.args.patch_len*5:border2 - border_start_AR + self.args.patch_len*5]
        best_pred_border_AE = best_pred_AE[border1 - border_start_AE - self.args.patch_len*5:border2 - border_start_AE + self.args.patch_len*5]
        best_pred_border_AR = best_pred_AR[border1 - border_start_AR - self.args.patch_len*5:border2 - border_start_AR + self.args.patch_len*5]
        data_path_AE = os.path.join(folder_path_AE, setting)
        data_path_AR = os.path.join(folder_path_AR, setting)
        file_path_border_AE = data_path_AE + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_border.pdf'
        file_path_AE = data_path_AE + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_testset.pdf'
        file_path_border_AR = data_path_AR + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_border.pdf'
        file_path_AR = data_path_AR + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_testset.pdf'
        
        
        if not os.path.exists(data_path_AE) or not os.path.exists(data_path_AR):
            os.makedirs(data_path_AE)
            os.makedirs(data_path_AR)

        visual_anomaly(input_border_AE, output_border_AE, best_pred_border_AE, self.args.patch_len*5, input_border_AE.shape[0]-self.args.patch_len*5, file_path_border_AE)
        visual_anomaly(input_AE, output_AE, best_pred_AE, border1-border_start_AE, border2-border_start_AE, file_path_AE)
        visual_anomaly(input_border_AR, output_border_AR, best_pred_border_AR, self.args.patch_len*5, input_border_AR.shape[0]-self.args.patch_len*5, file_path_border_AR)
        visual_anomaly(input_AR, output_AR, best_pred_AR, border1-border_start_AR, border2-border_start_AR, file_path_AR)
        
        def is_overlap(index):
            start = index * self.args.patch_len + border_start
            end = start + self.args.patch_len
            if border1 <= start <= border2 or border1 <= end <= border2 or start <= border1 and end >= border2:
                return True
            else:
                return False
            
        
        return
