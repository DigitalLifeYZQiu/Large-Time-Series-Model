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
from utils.masking import patch_mask, expand_tensor, noise_mask, patch_random_mask

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_AE(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_AE, self).__init__(args)

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
                        # print("batch_x:",batch_x.shape)
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print("mask_rate:",self.args.mask_rate)
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
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
                        B, T, N = batch_x.shape # (B,S,C)
                        # print("batch_x.shape:",batch_x.shape)
                        assert T % self.args.patch_len == 0

                        mask_rate = self.args.mask_rate
                        # mask_patch_len = 4
                        # mask_stride = 4
                        # print("T/self.args.patch_len:",T/self.args.patch_len)
                        reshaped_batch_x = batch_x.view(B, int(T/self.args.patch_len), self.args.patch_len, N)
                        reshaped_batch_x = reshaped_batch_x.permute(0, 1, 3, 2)
                        x_masked, x_kept, mask, ids_restore = patch_random_mask(reshaped_batch_x, mask_rate)
                        # mask = expand_tensor(mask, mask_patch_len)
                        # mask = mask.reshape(B, N, -1)[:, :, :T].permute(0, 2, 1)
                        inp = x_masked
                        inp = inp.permute(0, 1, 3, 2)
                        inp = inp.view(B, T, N)
                        # print("inp:",inp.shape)

                        outputs = self.model(inp[:, self.args.patch_len:, :], None, None, None, mask)
                        batch_x = batch_x[:, self.args.patch_len:, :]
                    else:
                        outputs = self.model(batch_x[:, self.args.patch_len:, :], None, None, None)
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
        if self.args.use_ims:
            status += '_ims'
        folder_path = './test_results/' + self.args.data + status +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        border_start = self.find_border_number(self.args.data_path)
        border1, border2 = self.find_border(self.args.data_path)

        token_count = 0
        if self.args.use_ims:
            rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len
        else:
            # rec_token_count = self.args.seq_len // self.args.patch_len
            rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len

        input_list = []
        output_list = []
        test_labels = []
        embedding_list = []
        feature_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruct the input sequence and record the loss as a sorted list
                if self.args.use_ims:
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)[:, :-self.args.patch_len, :]
                    batch_x = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                    batch_y = batch_y[:, self.args.patch_len:-self.args.patch_len]
                    # outputs = outputs[:, :-self.args.patch_len, :]
                    embeds = self.model.getEmbedding(batch_x)
                    features = self.model.getFeature(batch_x)
                else:
                    outputs = self.model(batch_x[:, self.args.patch_len:-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                    batch_y = batch_y[:, self.args.patch_len:-self.args.patch_len]
                    embeds = self.model.getEmbedding(batch_x)
                    features = self.model.getFeature(batch_x)
                input_list.append(batch_x[0, :, -1].detach().cpu().numpy())
                output_list.append(outputs[0, :, -1].detach().cpu().numpy())
                test_labels.append(batch_y.reshape(-1).detach().cpu().numpy())
                embedding_list.append(embeds.detach().cpu().numpy())
                feature_list.append(features.detach().cpu().numpy())

                score = self.anomaly_criterion(outputs, batch_x)
                score_list.append(score.detach().cpu().numpy()) 
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
        test_labels = np.concatenate(test_labels, axis=0).flatten()
        input = np.concatenate(input_list, axis=0).reshape(-1)
        output = np.concatenate(output_list, axis=0).reshape(-1)
        score_list = np.concatenate(score_list, axis=0).reshape(-1)
        

        
        # 输出adjustment best f1及best f1在最佳阈值下的原始结果
        best_pred_adj, best_pred = adjbestf1(test_labels, score_list, 100)
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
        
        accuracy = accuracy.astype(np.float32)
        recall = recall.astype(np.float32)
        f_score = f_score.astype(np.float32)
        adjaccuracy = adjaccuracy.astype(np.float32)
        adjprecision = adjprecision.astype(np.float32)
        adjrecall = adjrecall.astype(np.float32)
        adjf_score = adjf_score.astype(np.float32)
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

        csv_file = 'results.csv'


        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))

        print("Results appended to", csv_file)

        #* visualization
        # half_patch_len = self.args.patch_len // 2
        # input_border = input[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        # output_border = output[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        # if not self.args.use_ims:
        #     border_start = border_start - self.args.patch_len
        input_border = input[border1 - border_start - self.args.patch_len*5 : border2 - border_start + self.args.patch_len*5]
        output_border = output[border1 - border_start - self.args.patch_len*5:border2 - border_start + self.args.patch_len*5]
        best_pred_border = best_pred[border1 - border_start - self.args.patch_len*5:border2 - border_start + self.args.patch_len*5]
        data_path = os.path.join(folder_path, setting)
        if self.args.use_ims:
            file_path_border = data_path + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_ims_border.pdf'
            file_path = data_path + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_ims_testset.pdf'
        else:
            file_path_border = data_path + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_border.pdf'
            file_path = data_path + '/' + self.args.data_path[:self.args.data_path.find('.')]+ '_testset.pdf'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        visual_anomaly(input_border, output_border, best_pred_border, self.args.patch_len*5, input_border.shape[0]-self.args.patch_len*5, file_path_border)
        visual_anomaly(input, output, best_pred, border1-border_start, border2-border_start, file_path)
        
        def is_overlap(index):
            start = index * self.args.patch_len + border_start
            end = start + self.args.patch_len
            if border1 <= start <= border2 or border1 <= end <= border2 or start <= border1 and end >= border2:
                return True
            else:
                return False
            
        if self.args.show_embedding:
            # Gerneral embedding
            embedding = np.concatenate(embedding_list, axis=0) # (test data length, rec_token_count, d_model)
            # Patch-wise embedding
            embedding_in_patch = np.concatenate(embedding_list, axis=0).reshape(-1, embedding_list[0].shape[-1]) # (test data length * rec_token_count, d_model)

            label=[]
            for index in range(embedding_in_patch.shape[0]):
                if is_overlap(index):
                    label.append(1)
                else:
                    label.append(0)
            if self.args.use_PCA:
                visualization_PCA(
                    X=embedding_in_patch,
                    labels=np.array(label),
                    token_nums=2,
                    path=data_path, 
                    name=self.args.data_path[:self.args.data_path.find('.')]+'_embedding_PCA.pdf'
                )
            else:
                visualization(
                    X=embedding_in_patch,
                    labels=np.array(label),
                    token_nums=2,
                    perplexity=self.args.tsne_perplexity,
                    path=data_path, 
                    name=self.args.data_path[:self.args.data_path.find('.')]+f'_embedding_perplexity{self.args.tsne_perplexity}.pdf'
                )
        if self.args.show_feature:
            # Gerneral embedding
            feature = np.concatenate(feature_list, axis=0) # (test data length, rec_token_count, d_model)
            # Patch-wise embedding
            feature_in_patch = np.concatenate(feature_list, axis=0).reshape(-1, feature_list[0].shape[-1]) # (test data length * rec_token_count, d_model)
            label=[]
            for index in range(feature_in_patch.shape[0]):
                if is_overlap(index):
                    label.append(1)
                else:
                    label.append(0)
            if self.args.use_PCA:
                visualization_PCA(
                    X=feature_in_patch,
                    labels=np.array(label),
                    token_nums=2,
                    path=data_path, 
                    name=self.args.data_path[:self.args.data_path.find('.')]+'_feature_PCA.pdf'
                )
            else:
                visualization(
                    X=feature_in_patch,
                    labels=np.array(label),
                    token_nums=2,
                    perplexity=self.args.tsne_perplexity,
                    path=data_path, 
                    name=self.args.data_path[:self.args.data_path.find('.')]+f'_feature_perplexity{self.args.tsne_perplexity}.pdf'
                )
        if self.args.show_score:
            score_in_patch = score_list.reshape(-1, self.args.patch_len) # (test data length * rec_token_count, patch_len)
            label=[]
            for index in range(score_in_patch.shape[0]):
                if is_overlap(index):
                    label.append(1)
                else:
                    label.append(0)
            if self.args.use_PCA:
                visualization_PCA(
                    X=score_in_patch,
                    labels=np.array(label),
                    token_nums=2,
                    path=data_path, 
                    name=self.args.data_path[:self.args.data_path.find('.')]+'_score_PCA.pdf'
                )
            else:
                visualization(
                    X=score_in_patch,
                    labels=np.array(label),
                    token_nums=2,
                    perplexity=self.args.tsne_perplexity,
                    path=data_path, 
                    name=self.args.data_path[:self.args.data_path.find('.')]+f'_score_perplexity{self.args.tsne_perplexity}.pdf'
                )
        return
