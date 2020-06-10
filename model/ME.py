
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .module.resnet import fixup_resnet152


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		# only the best case
		self.light_conv_num = 6
		self.poly_order = 3
		self.conv_param_num = 12

		self.poly_num = int((self.poly_order + 1) * (self.poly_order + 2) * (self.poly_order + 3) / 6)
		feature_num = self.light_conv_num * self.conv_param_num + self.poly_num * 3

		self.meta = fixup_resnet152(out_ch=feature_num)

	def light_conv(self, x, param):
		batch_size = x.size()[0]
		kernel = param[:, 0:9].view(batch_size, 3, 3, 1, 1)
		bias = param[:, 9:12].view(batch_size, 3)

		x_b2c = torch.cat(torch.split(x, 1, dim=0), dim=1)
		kernel_b2c = torch.cat(torch.split(kernel, 1, dim=0), dim=1).squeeze_(0)
		bias_b2c = torch.cat(torch.split(bias, 1, dim=0), dim=1).squeeze_(0)

		output = F.conv2d(x_b2c, kernel_b2c, bias=bias_b2c, groups=batch_size)
		output = torch.cat(torch.split(output, 3, dim=1), dim=0)

		output = x + output

		output = F.leaky_relu(output, negative_slope=0.2, inplace=True)

		return output

	def att_map(self, coef, h_mat, w_mat, x_int):
		att = \
			coef[:, 0].view(-1, 1, 1, 1)*torch.pow(h_mat, 3) + \
			coef[:, 1].view(-1, 1, 1, 1)*torch.pow(h_mat, 2)*w_mat + \
			coef[:, 2].view(-1, 1, 1, 1)*torch.pow(h_mat, 2)*x_int + \
			coef[:, 3].view(-1, 1, 1, 1)*torch.pow(h_mat, 2) + \
			coef[:, 4].view(-1, 1, 1, 1)*h_mat*torch.pow(w_mat, 2) + \
			coef[:, 5].view(-1, 1, 1, 1)*h_mat*w_mat*x_int + \
			coef[:, 6].view(-1, 1, 1, 1)*h_mat*w_mat + \
			coef[:, 7].view(-1, 1, 1, 1)*h_mat*torch.pow(x_int, 2) + \
			coef[:, 8].view(-1, 1, 1, 1)*h_mat*x_int + \
			coef[:, 9].view(-1, 1, 1, 1)*h_mat + \
			coef[:, 10].view(-1, 1, 1, 1)*torch.pow(w_mat, 3) + \
			coef[:, 11].view(-1, 1, 1, 1)*torch.pow(w_mat, 2)*x_int + \
			coef[:, 12].view(-1, 1, 1, 1)*torch.pow(w_mat, 2) + \
			coef[:, 13].view(-1, 1, 1, 1)*w_mat*torch.pow(x_int, 2) + \
			coef[:, 14].view(-1, 1, 1, 1)*w_mat*x_int + \
			coef[:, 15].view(-1, 1, 1, 1)*w_mat + \
			coef[:, 16].view(-1, 1, 1, 1)*torch.pow(x_int, 3) + \
			coef[:, 17].view(-1, 1, 1, 1)*torch.pow(x_int, 2) + \
			coef[:, 18].view(-1, 1, 1, 1)*x_int + \
			coef[:, 19].view(-1, 1, 1, 1)

		return att

	def forward(self, x, thumb_x, position=None):
		batch_size, _, x_h, x_w = x.size()
		_, _, thumb_h, thumb_w = thumb_x.size()

		# feature
		feature = self.meta(thumb_x)

		# position
		if position == None:
			h_mat = torch.arange(x_h).view(1, 1, x_h, 1).repeat(1, 1, 1, x_w).cuda()
			h_mat = h_mat / float(x_h)

			w_mat = torch.arange(x_w).view(1, 1, 1, x_w).repeat(1, 1, x_h, 1).cuda()
			w_mat = w_mat / float(x_w)
		else:
			h_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+x_h).view(1, 1, x_h, 1).repeat(1, 1, 1, x_w),
				position[2])
			), dim=0).cuda() 

			h_mat = h_mat / (position[0]).float().view(batch_size, 1, 1, 1).expand_as(h_mat)

			w_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+x_w).view(1, 1, 1, x_w).repeat(1, 1, x_h, 1),
				position[3])
			), dim=0).cuda()

			w_mat = w_mat / (position[1]).float().view(batch_size, 1, 1, 1).expand_as(w_mat)
			
			thumb_h_mat = torch.arange(thumb_h).view(1, 1, thumb_h, 1).repeat(batch_size, 1, 1, thumb_w).cuda()
			thumb_h_mat = thumb_h_mat / float(thumb_h)

			thumb_w_mat = torch.arange(thumb_w).view(1, 1, 1, thumb_w).repeat(batch_size, 1, thumb_h, 1).cuda()
			thumb_w_mat = thumb_w_mat / float(thumb_w)

		# feature to parameter
		offset = 0

		params = []
		for i in range(self.light_conv_num):
			params.append(feature[:, (offset+i*self.conv_param_num):(offset+(i+1)*self.conv_param_num)])
		offset = offset+(i+1)*self.conv_param_num if self.light_conv_num > 0 else offset

		coef_R = feature[:, offset:offset+self.poly_num]
		coef_G = feature[:, offset+self.poly_num:offset+self.poly_num*2]
		coef_B = feature[:, offset+self.poly_num*2:offset+self.poly_num*3]

		# main process
		residual = x

		for i in range(self.light_conv_num):
			x = self.light_conv(x, params[i])

		x = residual + x

		x_R = x[:, [0], : ,:]
		x_G = x[:, [1], : ,:]
		x_B = x[:, [2], : ,:]

		att_R = self.att_map(coef_R, h_mat, w_mat, x_R)
		att_G = self.att_map(coef_G, h_mat, w_mat, x_G)
		att_B = self.att_map(coef_B, h_mat, w_mat, x_B)

		att = torch.cat([att_R, att_G, att_B], dim=1)

		x = x * (1 + att)

		if position == None:
			return x, thumb_x

		# thumbnail process
		residual_t = thumb_x

		for i in range(self.light_conv_num):
			thumb_x = self.light_conv(thumb_x, params[i])

		thumb_x = residual_t + thumb_x

		thumb_x_R = thumb_x[:, [0], : ,:]
		thumb_x_G = thumb_x[:, [1], : ,:]
		thumb_x_B = thumb_x[:, [2], : ,:]

		thumb_att_R = self.att_map(coef_R, thumb_h_mat, thumb_w_mat, thumb_x_R)
		thumb_att_G = self.att_map(coef_G, thumb_h_mat, thumb_w_mat, thumb_x_G)
		thumb_att_B = self.att_map(coef_B, thumb_h_mat, thumb_w_mat, thumb_x_B)

		thumb_att = torch.cat([thumb_att_R, thumb_att_G, thumb_att_B], dim=1)

		thumb_x = thumb_x * (1 + thumb_att)
		
		return x, thumb_x
