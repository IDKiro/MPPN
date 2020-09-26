
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .module.resnet import fixup_resnet18, fixup_resnet34, fixup_resnet50, fixup_resnet101, fixup_resnet152


class Network(nn.Module):
	def __init__(self, light_conv_num=4, poly_order=3, arch='resnet18'):
		super(Network, self).__init__()
		self.light_conv_num = light_conv_num
		self.poly_order = poly_order

		self.conv_param_num = 15
		self.poly_num = int((poly_order + 1) * (poly_order + 2) * (poly_order + 3) / 6)
		feature_num = light_conv_num * self.conv_param_num + self.poly_num * 3

		if arch.startswith('resnet18'):
			self.encoder = fixup_resnet18(out_ch=feature_num)
		elif arch.startswith('resnet34'):
			self.encoder = fixup_resnet34(out_ch=feature_num)
		elif arch.startswith('resnet50'):
			self.encoder = fixup_resnet50(out_ch=feature_num)
		elif arch.startswith('resnet101'):
			self.encoder = fixup_resnet101(out_ch=feature_num)
		elif arch.startswith('resnet152'):
			self.encoder = fixup_resnet152(out_ch=feature_num)
		else:
			raise Exception("This architecture has not been supported yet")

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

	def poly_att(self, x_int, h_mat, w_mat, coef):
		att = coef[:, [0], :, :]

		if self.poly_order >= 1:
			att = att + \
				coef[:, [1], :, :]*h_mat + \
				coef[:, [2], :, :]*w_mat + \
				coef[:, [3], :, :]*x_int

		if self.poly_order >= 2:
			att = att + \
				coef[:, [4], :, :]*torch.pow(h_mat, 2) + \
				coef[:, [5], :, :]*torch.pow(w_mat, 2) + \
				coef[:, [6], :, :]*torch.pow(x_int, 2) + \
				coef[:, [7], :, :]*h_mat*w_mat + \
				coef[:, [8], :, :]*h_mat*x_int + \
				coef[:, [9], :, :]*w_mat*x_int

		if self.poly_order >= 3:
			att = att + \
				coef[:, [10], :, :]*torch.pow(h_mat, 3) + \
				coef[:, [11], :, :]*torch.pow(w_mat, 3) + \
				coef[:, [12], :, :]*torch.pow(x_int, 3) + \
				coef[:, [13], :, :]*torch.pow(h_mat, 2)*w_mat + \
				coef[:, [14], :, :]*torch.pow(h_mat, 2)*x_int + \
				coef[:, [15], :, :]*torch.pow(w_mat, 2)*x_int + \
				coef[:, [16], :, :]*h_mat*torch.pow(w_mat, 2) + \
				coef[:, [17], :, :]*h_mat*torch.pow(x_int, 2) + \
				coef[:, [18], :, :]*w_mat*torch.pow(x_int, 2) + \
				coef[:, [19], :, :]*h_mat*w_mat*x_int

		if self.poly_order >= 4:
			att = att + \
				coef[:, [20], :, :]*torch.pow(h_mat, 4) + \
				coef[:, [21], :, :]*torch.pow(w_mat, 4) + \
				coef[:, [22], :, :]*torch.pow(x_int, 4) + \
				coef[:, [23], :, :]*torch.pow(h_mat, 3)*w_mat + \
				coef[:, [24], :, :]*torch.pow(h_mat, 3)*x_int + \
				coef[:, [25], :, :]*torch.pow(w_mat, 3)*x_int + \
				coef[:, [26], :, :]*torch.pow(h_mat, 2)*torch.pow(w_mat, 2) + \
				coef[:, [27], :, :]*torch.pow(h_mat, 2)*torch.pow(x_int, 2) + \
				coef[:, [28], :, :]*torch.pow(w_mat, 2)*torch.pow(x_int, 2) + \
				coef[:, [29], :, :]*h_mat*torch.pow(w_mat, 3) + \
				coef[:, [30], :, :]*h_mat*torch.pow(x_int, 3) + \
				coef[:, [31], :, :]*w_mat*torch.pow(x_int, 3) + \
				coef[:, [32], :, :]*torch.pow(h_mat, 2)*w_mat*x_int + \
				coef[:, [33], :, :]*h_mat*torch.pow(w_mat, 2)*x_int + \
				coef[:, [34], :, :]*h_mat*w_mat*torch.pow(x_int, 2)

		if self.poly_order >= 5:
			att = att + \
				coef[:, [35], :, :]*torch.pow(h_mat, 5) + \
				coef[:, [36], :, :]*torch.pow(w_mat, 5) + \
				coef[:, [37], :, :]*torch.pow(x_int, 5) + \
				coef[:, [38], :, :]*torch.pow(h_mat, 4)*w_mat + \
				coef[:, [39], :, :]*torch.pow(h_mat, 4)*x_int + \
				coef[:, [40], :, :]*torch.pow(w_mat, 4)*x_int + \
				coef[:, [41], :, :]*torch.pow(h_mat, 3)*torch.pow(w_mat, 2) + \
				coef[:, [42], :, :]*torch.pow(h_mat, 3)*torch.pow(x_int, 2) + \
				coef[:, [43], :, :]*torch.pow(w_mat, 3)*torch.pow(x_int, 2) + \
				coef[:, [44], :, :]*torch.pow(h_mat, 2)*torch.pow(w_mat, 3) + \
				coef[:, [45], :, :]*torch.pow(h_mat, 2)*torch.pow(x_int, 3) + \
				coef[:, [46], :, :]*torch.pow(w_mat, 2)*torch.pow(x_int, 3) + \
				coef[:, [47], :, :]*h_mat*torch.pow(w_mat, 4) + \
				coef[:, [48], :, :]*h_mat*torch.pow(x_int, 4) + \
				coef[:, [49], :, :]*w_mat*torch.pow(x_int, 4) + \
				coef[:, [50], :, :]*torch.pow(h_mat, 3)*w_mat*x_int + \
				coef[:, [51], :, :]*h_mat*torch.pow(w_mat, 3)*x_int + \
				coef[:, [52], :, :]*h_mat*w_mat*torch.pow(x_int, 3) + \
				coef[:, [53], :, :]*torch.pow(h_mat, 2)*torch.pow(w_mat, 2)*x_int + \
				coef[:, [54], :, :]*torch.pow(h_mat, 2)*w_mat*torch.pow(x_int, 2) + \
				coef[:, [55], :, :]*h_mat*torch.pow(w_mat, 2)*torch.pow(x_int, 2)

		if self.poly_order > 5:
			raise Exception("Higher order has not been supported yet")

		output = x_int + att * x_int
		return output

	def forward(self, x, thumb_x, position=None):
		batch_size, _, x_h, x_w = x.size()
		_, _, thumb_h, thumb_w = thumb_x.size()

		# feature
		feature = self.encoder(thumb_x)

		# position
		if position == None:
			h_mat = torch.arange(x_h).view(1, 1, x_h, 1).repeat(1, 1, 1, x_w).cuda()
			h_mat = h_mat / float(x_h - 1) * 2. - 1.

			w_mat = torch.arange(x_w).view(1, 1, 1, x_w).repeat(1, 1, x_h, 1).cuda()
			w_mat = w_mat / float(x_w - 1) * 2. - 1.
		else:
			h_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+x_h).view(1, 1, x_h, 1).repeat(1, 1, 1, x_w),
				position[2])
			), dim=0).cuda() 

			h_mat = h_mat / (position[0] - 1).float().view(batch_size, 1, 1, 1).expand_as(h_mat) * 2. - 1.

			w_mat = torch.cat(list(map(
				lambda s: torch.arange(start=s, end=s+x_w).view(1, 1, 1, x_w).repeat(1, 1, x_h, 1),
				position[3])
			), dim=0).cuda()

			w_mat = w_mat / (position[1] - 1).float().view(batch_size, 1, 1, 1).expand_as(w_mat) * 2. - 1.
			
			thumb_h_mat = torch.arange(thumb_h).view(1, 1, thumb_h, 1).repeat(batch_size, 1, 1, thumb_w).cuda()
			thumb_h_mat = thumb_h_mat / float(thumb_h - 1) * 2. - 1.

			thumb_w_mat = torch.arange(thumb_w).view(1, 1, 1, thumb_w).repeat(batch_size, 1, thumb_h, 1).cuda()
			thumb_w_mat = thumb_w_mat / float(thumb_w - 1) * 2. - 1.

		# feature to parameter
		offset = 0

		params = []
		for i in range(self.light_conv_num):
			params.append(feature[:, (offset+i*self.conv_param_num):(offset+(i+1)*self.conv_param_num)])
		offset = offset+(i+1)*self.conv_param_num if self.light_conv_num > 0 else offset

		coef_R = feature[:, offset:offset+self.poly_num].unsqueeze_(2).unsqueeze_(3)
		coef_G = feature[:, offset+self.poly_num:offset+self.poly_num*2].unsqueeze_(2).unsqueeze_(3)
		coef_B = feature[:, offset+self.poly_num*2:offset+self.poly_num*3].unsqueeze_(2).unsqueeze_(3)

		# main process
		for i in range(self.light_conv_num):
			x = self.light_conv(x, params[i])

		x = torch.cat([
				self.poly_att(x[:, [0], : ,:], h_mat, w_mat, coef_R),
				self.poly_att(x[:, [1], : ,:], h_mat, w_mat, coef_G), 
				self.poly_att(x[:, [2], : ,:], h_mat, w_mat, coef_B)
				], dim=1)

		if position == None:
			return x

		# thumbnail process
		for i in range(self.light_conv_num):
			thumb_x = self.light_conv(thumb_x, params[i])

		thumb_x = torch.cat([
				self.poly_att(thumb_x[:, [0], : ,:], thumb_h_mat, thumb_w_mat, coef_R),
				self.poly_att(thumb_x[:, [1], : ,:], thumb_h_mat, thumb_w_mat, coef_G),
				self.poly_att(thumb_x[:, [2], : ,:], thumb_h_mat, thumb_w_mat, coef_B)
			], dim=1)

		return x, thumb_x
