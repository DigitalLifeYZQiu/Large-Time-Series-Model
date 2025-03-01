import torch
import torch.nn as nn
import torch.nn.functional as F
import darts
from darts.models import ARIMA
from darts import TimeSeries
import numpy as np
from data_provider.data_factory import data_provider


class Model(nn.Module):
	"""
	Autoformer is the first method to achieve the series-wise connection,
	with inherent O(LlogL) complexity
	Paper link: https://openreview.net/pdf?id=I55UqU-M11y
	"""
	
	def __init__(self, configs):
		super(Model, self).__init__()
		self.task_name = configs.task_name
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.backbone = ARIMA()
		# self.fit(configs)
	
	def fit(self, configs):
		self.train_data, self.train_loader = data_provider(configs, flag='train')
		trainset = self.train_data.data_x.squeeze()
		print(f"{configs.model} Fitting Training Set")
		self.backbone.fit(TimeSeries.from_values(trainset))
	
	def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
		x_enc = x_enc.to('cpu').detach().numpy()
		B, S, C = x_enc.shape
		all_preds = []
		for b in range(B):
			preds_for_b = []
			for c in range(C):
				data_series = TimeSeries.from_values(x_enc[b, :, c])
				self.backbone.fit(data_series)
				pred = self.backbone.predict(self.pred_len).values()
				pred = pred.reshape(1, self.pred_len, 1)
				preds_for_b.append(pred)
			preds_for_b = np.concatenate(preds_for_b, axis=2)
			all_preds.append(preds_for_b)
		final_preds = np.concatenate(all_preds, axis=0)
		final_preds = final_preds.reshape((B, self.pred_len, C))
		return torch.Tensor(final_preds)
	
	def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
		raise NotImplementedError
	
	def anomaly_detection(self, x_enc):
		raise NotImplementedError
	
	def classification(self, x_enc, x_mark_enc):
		raise NotImplementedError
	
	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		# if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
		if 'forecast' in self.task_name:
			dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
			return dec_out[:, -self.pred_len:, :]  # [B, L, D]
		if self.task_name == 'imputation':
			raise NotImplementedError
		if self.task_name == 'anomaly_detection':
			raise NotImplementedError
		if self.task_name == 'classification':
			raise NotImplementedError
		return None