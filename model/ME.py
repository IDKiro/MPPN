
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .module.resnet import fixup_resnet152 as meta


class Network(nn.Module):
	def __init__(self, conv3_num=3, conv1_num=5, poly_num=4):
		super(Network, self).__init__()
		self.conv3_num = conv3_num
		self.conv1_num = conv1_num
		self.poly_num = poly_num
		self.meta = meta(out_ch=conv3_num*84+conv1_num*12+poly_num*12)

	def conv3x3(self, x, feature):
		batch_size = x.size()[0]
		kernel = feature[:, 0:81].view(batch_size, 3, 3, 3, 3)
		bias = feature[:, 81:84].view(batch_size, 3)

		input_list = list(torch.split(x, 1, dim=0))
		kernel_list = list(torch.split(kernel, 1, dim=0))
		bias_list = list(torch.split(bias, 1, dim=0))

		output_list = list(map(
			lambda x, y, z: F.conv2d(x, torch.squeeze(y, dim=0), bias=torch.squeeze(z, dim=0), padding=1), 
			input_list, kernel_list, bias_list)
		)
			
		output = torch.cat(output_list, dim=0)

		output = F.leaky_relu(output, negative_slope=0.2, inplace=True)

		return output

	def conv1x1(self, x, feature):
		batch_size = x.size()[0]
		kernel = feature[:, 0:9].view(batch_size, 3, 3, 1, 1)
		bias = feature[:, 9:12].view(batch_size, 3)

		input_list = list(torch.split(x, 1, dim=0))
		kernel_list = list(torch.split(kernel, 1, dim=0))
		bias_list = list(torch.split(bias, 1, dim=0))

		output_list = list(map(
			lambda x, y, z: F.conv2d(x, torch.squeeze(y, dim=0), bias=torch.squeeze(z, dim=0)), 
			input_list, kernel_list, bias_list)
		)
			
		output = torch.cat(output_list, dim=0)

		output = F.leaky_relu(output, negative_slope=0.2, inplace=True)

		return output

	def att_map(self, feature, h_mat, w_mat, x_int):
		att = torch.ones_like(x_int)

		for i in range(self.poly_num):
			att = att * \
			(feature[:, i*4+0].view(-1, 1, 1, 1).expand_as(x_int)*h_mat + \
			feature[:, i*4+1].view(-1, 1, 1, 1).expand_as(x_int)*w_mat + \
			feature[:, i*4+2].view(-1, 1, 1, 1).expand_as(x_int)*x_int + \
			feature[:, i*4+3].view(-1, 1, 1, 1).expand_as(x_int))

		return att

	def forward(self, x, thumb_x, location=None):
		batch_size, _, x_h, x_w = x.size()
		_, _, thumb_h, thumb_w = thumb_x.size()

		feature = self.meta(thumb_x)
		
		# For main
		residual = x

		for i in range(self.conv3_num):
			x = self.conv3x3(x, feature[:, i*84:(i+1)*84])
		bias = (i+1)*84

		for i in range(self.conv1_num):
			x = self.conv1x1(x, feature[:, (bias+i*12):(bias+(i+1)*12)])
		bias = bias+(i+1)*12

		x = residual + x

		if location == None:
			h_mat = torch.arange(x_h).view(1, 1, x_h, 1).repeat(1, 1, 1, x_w).cuda()
			h_mat = h_mat / float(x_h)

			w_mat = torch.arange(x_w).view(1, 1, 1, x_w).repeat(1, 1, x_h, 1).cuda()
			w_mat = w_mat / float(x_w)
		else:
			h_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+x_h).view(1, 1, x_h, 1).repeat(1, 1, 1, x_w),
				location[2])
			), dim=0).cuda() 

			h_mat = h_mat / location[0].float().view(batch_size, 1, 1, 1).expand_as(h_mat)

			w_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+x_w).view(1, 1, 1, x_w).repeat(1, 1, x_h, 1),
				location[3])
			), dim=0).cuda()

			w_mat = w_mat / location[1].float().view(batch_size, 1, 1, 1).expand_as(w_mat)

		feature_R = feature[:, bias:bias+self.poly_num*4]
		feature_G = feature[:, bias+self.poly_num*4:bias+self.poly_num*8]
		feature_B = feature[:, bias+self.poly_num*8:bias+self.poly_num*12]

		x_R = x[:, [0], : ,:]
		x_G = x[:, [1], : ,:]
		x_B = x[:, [2], : ,:]

		att_R = self.att_map(feature_R, h_mat, w_mat, x_R)
		att_G = self.att_map(feature_G, h_mat, w_mat, x_G)
		att_B = self.att_map(feature_B, h_mat, w_mat, x_B)

		att = torch.cat([att_R, att_G, att_B], dim=1)

		x = x * (1 + att)

		# For thumbnail
		bias = 0
		residual_t = thumb_x

		for i in range(self.conv3_num):
			thumb_x = self.conv3x3(thumb_x, feature[:, i*84:(i+1)*84])
		bias = (i+1)*84

		for i in range(self.conv1_num):
			thumb_x = self.conv1x1(thumb_x, feature[:, (bias+i*12):(bias+(i+1)*12)])
		bias = bias+(i+1)*12

		thumb_x = residual_t + thumb_x

		thumb_h_mat = torch.arange(thumb_h).view(1, 1, thumb_h, 1).repeat(1, 1, 1, thumb_w).cuda()
		thumb_h_mat = thumb_h_mat / float(thumb_h)

		thumb_w_mat = torch.arange(thumb_w).view(1, 1, 1, thumb_w).repeat(1, 1, thumb_h, 1).cuda()
		thumb_w_mat = thumb_w_mat / float(thumb_w)

		thumb_x_R = thumb_x[:, [0], : ,:]
		thumb_x_G = thumb_x[:, [1], : ,:]
		thumb_x_B = thumb_x[:, [2], : ,:]

		thumb_att_R = self.att_map(feature_R, thumb_h_mat, thumb_w_mat, thumb_x_R)
		thumb_att_G = self.att_map(feature_G, thumb_h_mat, thumb_w_mat, thumb_x_G)
		thumb_att_B = self.att_map(feature_B, thumb_h_mat, thumb_w_mat, thumb_x_B)

		thumb_att = torch.cat([thumb_att_R, thumb_att_G, thumb_att_B], dim=1)

		thumb_x = thumb_x * (1 + thumb_att)

		return x, thumb_x