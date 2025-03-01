import os
import time
import warnings
import csv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
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
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    pred = outputs[:, -self.args.seq_len:, :]
                    true = batch_y
                    if flag == 'vali':
                        loss = criterion(pred, true)
                    elif flag == 'test':  # in this case, only pred_len is used to calculate loss
                        pred = pred[:, -self.args.pred_len:, :]
                        true = true[:, -self.args.pred_len:, :]
                        loss = criterion(pred, true)
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def finetune(self, setting):
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)


        for epoch in range(self.args.finetune_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    # only use the forecast window to calculate loss
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss_val += loss
                count += 1

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
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # def test(self, setting, test=0):
    #
    #     print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
    #     attns = []
    #     folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
    #     if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
    #         os.makedirs(folder_path)
    #     self.model.eval()
    #     if self.args.output_len_list is None:
    #         self.args.output_len_list = [self.args.output_len]
    #
    #     preds_list = [[] for _ in range(len(self.args.output_len_list))]
    #     trues_list = [[] for _ in range(len(self.args.output_len_list))]
    #     self.args.output_len_list.sort()
    #
    #     with torch.no_grad():
    #         for output_ptr in range(len(self.args.output_len_list)):
    #             self.args.output_len = self.args.output_len_list[output_ptr]
    #             test_data, test_loader = data_provider(self.args, flag='test')
    #             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #                 batch_x = batch_x.float().to(self.device)
    #                 batch_y = batch_y.float().to(self.device)
    #
    #                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #                 inference_steps = self.args.output_len // self.args.pred_len
    #                 dis = self.args.output_len - inference_steps * self.args.pred_len
    #                 if dis != 0:
    #                     inference_steps += 1
    #                 pred_y = []
    #                 for j in range(inference_steps):
    #                     if len(pred_y) != 0:
    #                         batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
    #                         tmp = batch_y_mark[:, j - 1:j, :]
    #                         batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
    #
    #                     if self.args.output_attention:
    #                         outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    #                     f_dim = -1 if self.args.features == 'MS' else 0
    #                     pred_y.append(outputs[:, -self.args.pred_len:, :])
    #                 pred_y = torch.cat(pred_y, dim=1)
    #
    #                 if dis != 0:
    #                     pred_y = pred_y[:, :-self.args.pred_len+dis, :]
    #
    #                 if self.args.use_ims:
    #                     batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
    #                         self.device)
    #                 else:
    #                     batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
    #                 outputs = pred_y.detach().cpu()
    #                 batch_y = batch_y.detach().cpu()
    #
    #                 if test_data.scale and self.args.inverse:
    #                     shape = outputs.shape
    #                     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
    #                     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
    #
    #                 outputs = outputs[:, :, f_dim:]
    #                 batch_y = batch_y[:, :, f_dim:]
    #
    #                 pred = outputs
    #                 true = batch_y
    #
    #                 preds_list[output_ptr].append(pred)
    #                 trues_list[output_ptr].append(true)
    #                 if i % 10 == 0:
    #                     input = batch_x.detach().cpu().numpy()
    #                     gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
    #                     pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
    #
    #                     if self.args.local_rank == 0:
    #                         if self.args.output_attention:
    #                             attn = attns[0].cpu().numpy()[0, 0, :, :]
    #                             attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))
    #
    #                         visual(gt, pd, os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))
    #
    #     if self.args.output_len_list is not None:
    #         for i in range(len(preds_list)):
    #             preds = preds_list[i]
    #             trues = trues_list[i]
    #             preds = torch.cat(preds, dim=0).numpy()
    #             trues = torch.cat(trues, dim=0).numpy()
    #             mae, mse, rmse, mape, mspe = metric(preds, trues)
    #             print(f"output_len: {self.args.output_len_list[i]}")
    #
    #             print('mse:{}, mae:{}'.format(mse, mae))
    #             f = open("result_long_term_forecast.txt", 'a')
    #             f.write(setting + "  \n")
    #             f.write('mse:{}, mae:{}'.format(mse, mae))
    #             f.write('\n')
    #             f.write('\n')
    #             f.close()
    #
    #     return
    
    def test(self, setting, test=0):
        if not self.args.is_training and self.args.finetuned_ckpt_path:
            print('loading model: ', self.args.finetuned_ckpt_path)
            if self.args.finetuned_ckpt_path.endswith('.pth'):
                sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")
                self.model.load_state_dict(sd, strict=True)

            elif self.args.finetuned_ckpt_path.endswith('.ckpt'):
                if self.args.use_multi_gpu:
                    sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")["state_dict"]
                    sd = {'module.' + k: v for k, v in sd.items()}
                    self.model.load_state_dict(sd, strict=True)
                else:
                    sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")["state_dict"]
                    self.model.load_state_dict(sd, strict=True)
            else:
                raise NotImplementedError

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        target_root_path = self.args.root_path
        target_data_path = self.args.data_path
        target_data = self.args.data
        test_data, test_loader = data_provider(self.args, flag='test')
        if isinstance(test_data, list) and isinstance(test_loader, list):
            results = []
            folder_path = './test_results/' + setting + '/' + target_data + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for data, loader in zip(test_data, test_loader):
                print("=====================Testing: {}=====================".format(data.root_path))
                print("=====================Demo: {}=====================".format(data.root_path))
                
                preds = []
                trues = []
                
                self.model.eval()
                
                with torch.no_grad():
                    # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mean, batch_std) in enumerate(loader):
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        
                        if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                            batch_x_mark = None
                            batch_y_mark = None
                        else:
                            batch_x_mark = batch_x_mark.float().to(self.device)
                            batch_y_mark = batch_y_mark.float().to(self.device)
                        
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        inference_steps = self.args.output_len // self.args.pred_len
                        dis = self.args.output_len - inference_steps * self.args.pred_len
                        if dis != 0:
                            inference_steps += 1
                        pred_y = []
                        # encoder - decoder
                        for j in range(inference_steps):
                            # import pdb; pdb.set_trace()
                            if len(pred_y) != 0:
                                batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                                tmp = batch_y_mark[:, j - 1:j, :]
                                batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                            
                            if self.args.use_amp:
                                with torch.cuda.amp.autocast():
                                    if self.args.output_attention:
                                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                    else:
                                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    # import pdb; pdb.set_trace()
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                            outputs = outputs[:, -self.args.pred_len:, -1:]
                            f_dim = -1 if self.args.features == 'MS' else 0
                            pred_y.append(outputs[:, -self.args.pred_len:, :])
                        pred_y = torch.cat(pred_y, dim=1)
                        if dis != 0:
                            pred_y = pred_y[:, :-dis, :]
                        
                        # batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(self.device)
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        outputs = pred_y[:, -self.args.pred_len:, :].detach().cpu()
                        batch_y = batch_y.detach().cpu()
                        B,S,C = outputs.shape
                        # outputs = outputs.reshape(B, -1)
                        # outputs = outputs * batch_std.reshape((-1,1)) + batch_mean.reshape((-1,1))
                        # outputs.reshape(B, S, C)
                        # outputs = torch.Tensor(data.scaler.transform(outputs.reshape(-1,1)).reshape((B,S,C)))
                        # batch_y = torch.Tensor(data.scaler.transform(batch_y.reshape(-1,1)).reshape((B,S,C)))
                        if data.scale and self.args.inverse:
                            shape = outputs.shape
                            outputs = data.inverse_transform(outputs.reshape(-1,1)).reshape(shape)
                            batch_y = data.inverse_transform(batch_y.reshape(-1,1)).reshape(shape)
                            outputs = data.test_scaler.transform(outputs.reshape(-1,1)).reshape(shape)
                            batch_y = data.test_scaler.transform(batch_y.reshape(-1,1)).reshape(shape)
                            outputs = torch.Tensor(outputs)
                            batch_y = torch.Tensor(batch_y)
                        
                        outputs = outputs[:, :, f_dim:]
                        batch_y = batch_y[:, :, f_dim:]
                        
                        pred = outputs
                        true = batch_y
                        
                        preds.append(pred)
                        trues.append(true)
                        
                        # if i % 10 == 0:
                        #     input = batch_x.detach().cpu().numpy()
                        #     gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        #     pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                        #
                        #     dir_path = folder_path + f'{self.args.output_len}/'
                        #     if not os.path.exists(dir_path):
                        #         os.makedirs(dir_path)
                        #     # np.save(os.path.join(dir_path, f'gt_{i}.npy'), np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0))
                        #     # np.save(os.path.join(dir_path, f'pd_{i}.npy'), np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0))
                        #     np.savez(os.path.join(dir_path, f'res_{i}.npz'), groundtruth=gt, predict=pd)
                        #     print(os.path.join(dir_path, f'res_{i}.npz'), "saved")
                        
                        # if self.args.use_multi_gpu:
                        #     visual(gt, pd, os.path.join(dir_path, f'{i}_{self.args.local_rank}.pdf'))
                        # else:
                        #     visual(gt, pd, os.path.join(dir_path, f'{i}_.pdf'))
                preds = torch.cat(preds, dim=0).numpy()
                trues = torch.cat(trues, dim=0).numpy()
                
                # # result save
                # folder_path = './results/' + setting + '/'
                # if not os.path.exists(folder_path):
                #     os.makedirs(folder_path)
                
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                results.append({'ckpt': self.args.ckpt_path, 'dataset': data.dataset_name, 'pred_len': self.args.pred_len, 'mse': mse, 'mae': mae})
                print('mse:{}, mae:{}'.format(mse, mae))
            print(f"results={results}")
            if len(results)>0:
                fieldnames = list(results[0].keys())
                with open(f"{folder_path}results.csv", 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # 写入列名
                    writer.writeheader()
                    
                    # 写入每一组实验结果
                    for result in results:
                        writer.writerow(result)
                    print(f'all results saved to {folder_path}results.csv')
                    
        else:
            print("=====================Testing: {}=====================".format(target_root_path + target_data_path))
            print("=====================Demo: {}=====================".format(target_root_path + target_data_path))

            preds = []
            trues = []
            folder_path = './test_results/' + setting + '/' + target_data_path + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            self.model.eval()

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    inference_steps = self.args.output_len // self.args.pred_len
                    dis = self.args.output_len - inference_steps * self.args.pred_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    # encoder - decoder
                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            tmp = batch_y_mark[:, j - 1:j, :]
                            batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, -1:]
                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred_y.append(outputs[:, -self.args.pred_len:, :])
                    pred_y = torch.cat(pred_y, dim=1)
                    if dis != 0:
                        pred_y = pred_y[:, :-dis, :]

                    # batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(self.device)
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = pred_y[:, -self.args.pred_len:, :].detach().cpu()
                    batch_y = batch_y.detach().cpu()

                    # if test_data.scale and self.args.inverse:
                    #     shape = outputs.shape
                    #     outputs = test_data.post_process(outputs.squeeze(0)).reshape(shape)
                    #     batch_y = test_data.post_process(batch_y.squeeze(0)).reshape(shape)
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)

                    # if i % 10 == 0:
                    #     input = batch_x.detach().cpu().numpy()
                    #     gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                    #     pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                    #
                    #     dir_path = folder_path + f'{self.args.output_len}/'
                    #     if not os.path.exists(dir_path):
                    #         os.makedirs(dir_path)
                    #     # np.save(os.path.join(dir_path, f'gt_{i}.npy'), np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0))
                    #     # np.save(os.path.join(dir_path, f'pd_{i}.npy'), np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0))
                    #     np.savez(os.path.join(dir_path, f'res_{i}.npz'), groundtruth=gt, predict=pd)
                    #     print(os.path.join(dir_path, f'res_{i}.npz'), "saved")

                        # if self.args.use_multi_gpu:
                        #     visual(gt, pd, os.path.join(dir_path, f'{i}_{self.args.local_rank}.pdf'))
                        # else:
                        #     visual(gt, pd, os.path.join(dir_path, f'{i}_.pdf'))
            preds = torch.cat(preds, dim=0).numpy()
            trues = torch.cat(trues, dim=0).numpy()

            # # result save
            # folder_path = './results/' + setting + '/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))

        return
