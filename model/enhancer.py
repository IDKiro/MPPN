
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import ResNet


class Network(nn.Module):
	def __init__(self, light_conv_num=4, poly_order=3):
		super(Network, self).__init__()
		self.light_conv_num = light_conv_num
		self.poly_order = poly_order

		self.conv_param_num = 15
		self.poly_num = int((poly_order + 1) * (poly_order + 2) * (poly_order + 3) / 6)
		feature_num = light_conv_num * self.conv_param_num + self.poly_num * 3

		self.index_conv = list(range(0, light_conv_num * self.conv_param_num + 1, self.conv_param_num))
		self.index_coef = list(range(0, self.poly_num * 3 + 1, 3))

		self.encoder = ResNet(planes=64, out_ch=feature_num)

	def light_conv(self, x, param):
		batch_size = x.size()[0]
		kernel = param[:, 0:9].view(batch_size, 3, 3, 1, 1)
		bias = param[:, 9:12].view(batch_size, 3)
		slope = param[:, 12:15].view(batch_size, 3, 1, 1)

		x_b2c = torch.cat(torch.split(x, 1, dim=0), dim=1)
		kernel_b2c = torch.cat(torch.split(kernel, 1, dim=0), dim=1).squeeze_(0)
		bias_b2c = torch.cat(torch.split(bias, 1, dim=0), dim=1).squeeze_(0)

		output = F.conv2d(x_b2c, kernel_b2c, bias=bias_b2c, groups=batch_size)
		output = torch.cat(torch.split(output, 3, dim=1), dim=0)

		output = x + output

		output = torch.max(output, torch.zeros_like(output)) + torch.min(output, torch.zeros_like(output)) * slope

		return output

	def get_loc(self, position):
		if position == None:
			h_mat = torch.arange(self.x_h).view(1, 1, self.x_h, 1).repeat(1, 1, 1, self.x_w).cuda()
			h_mat = h_mat / float(self.x_h - 1) * 2. - 1.

			w_mat = torch.arange(self.x_w).view(1, 1, 1, self.x_w).repeat(1, 1, self.x_h, 1).cuda()
			w_mat = w_mat / float(self.x_w - 1) * 2. - 1.
			
			return h_mat, w_mat, None, None
		else:
			h_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+self.x_h).view(1, 1, self.x_h, 1).repeat(1, 1, 1, self.x_w),
				position[2])
			), dim=0).cuda() 
			h_mat = h_mat / (position[0] - 1).float().view(self.batch_size, 1, 1, 1).expand_as(h_mat) * 2. - 1.

			w_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+self.x_w).view(1, 1, 1, self.x_w).repeat(1, 1, self.x_h, 1),
				position[3])
			), dim=0).cuda()
			w_mat = w_mat / (position[1] - 1).float().view(self.batch_size, 1, 1, 1).expand_as(w_mat) * 2. - 1.
			
			t_h_mat = torch.arange(self.t_h).view(1, 1, self.t_h, 1).repeat(self.batch_size, 1, 1, self.t_w).cuda()
			t_h_mat = t_h_mat / float(self.t_h - 1) * 2. - 1.

			t_w_mat = torch.arange(self.t_w).view(1, 1, 1, self.t_w).repeat(self.batch_size, 1, self.t_h, 1).cuda()
			t_w_mat = t_w_mat / float(self.t_w - 1) * 2. - 1.

			return h_mat, w_mat, t_h_mat, t_w_mat

	def poly_att(self, x, h_mat, w_mat, coef):
		att = coef[:, self.index_coef[0]:self.index_coef[1], :, :]

		if self.poly_order >= 1:
			att = att + \
				coef[:, self.index_coef[1]:self.index_coef[2], :, :]*h_mat + \
				coef[:, self.index_coef[2]:self.index_coef[3], :, :]*w_mat + \
				coef[:, self.index_coef[3]:self.index_coef[4], :, :]*x

		if self.poly_order >= 2:
			att = att + \
				coef[:, self.index_coef[4]:self.index_coef[5], :, :]*h_mat*h_mat + \
				coef[:, self.index_coef[5]:self.index_coef[6], :, :]*w_mat*w_mat + \
				coef[:, self.index_coef[6]:self.index_coef[7], :, :]*x*x + \
				coef[:, self.index_coef[7]:self.index_coef[8], :, :]*h_mat*w_mat + \
				coef[:, self.index_coef[8]:self.index_coef[9], :, :]*h_mat*x + \
				coef[:, self.index_coef[9]:self.index_coef[10], :, :]*w_mat*x

		if self.poly_order >= 3:
			att = att + \
				coef[:, self.index_coef[10]:self.index_coef[11], :, :]*h_mat*h_mat*h_mat + \
				coef[:, self.index_coef[11]:self.index_coef[12], :, :]*w_mat*w_mat*w_mat + \
				coef[:, self.index_coef[12]:self.index_coef[13], :, :]*x*x*x + \
				coef[:, self.index_coef[13]:self.index_coef[14], :, :]*h_mat*h_mat*w_mat + \
				coef[:, self.index_coef[14]:self.index_coef[15], :, :]*h_mat*h_mat*x + \
				coef[:, self.index_coef[15]:self.index_coef[16], :, :]*w_mat*w_mat*x + \
				coef[:, self.index_coef[16]:self.index_coef[17], :, :]*h_mat*w_mat*w_mat + \
				coef[:, self.index_coef[17]:self.index_coef[18], :, :]*h_mat*x*x + \
				coef[:, self.index_coef[18]:self.index_coef[19], :, :]*w_mat*x*x + \
				coef[:, self.index_coef[19]:self.index_coef[20], :, :]*h_mat*w_mat*x

		if self.poly_order > 3:
			raise Exception("Higher order has not been supported yet")

		return att * x + x

	def forward(self, x, t_x, position=None):
		self.batch_size, _, self.x_h, self.x_w = x.size()
		_, _, self.t_h, self.t_w = t_x.size()

		# feature
		feature = self.encoder(t_x)

		# position
		h_mat, w_mat, t_h_mat, t_w_mat = self.get_loc(position)

		# feature to parameter
		params = []
		for i in range(self.light_conv_num):
			params.append(feature[:, self.index_conv[i]:self.index_conv[i+1]])

		coef = feature[:, self.index_conv[-1]:].unsqueeze_(2).unsqueeze_(3)

		# main process
		for i in range(self.light_conv_num):
			x = self.light_conv(x, params[i])

		x = self.poly_att(x, h_mat, w_mat, coef)

		if position == None:
			return x

		# thumbnail process
		for i in range(self.light_conv_num):
			t_x = self.light_conv(t_x, params[i])

		t_x = self.poly_att(t_x, t_h_mat, t_w_mat, coef)

		return x, t_x
