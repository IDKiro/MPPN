from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim

from model.ME import Network as ME
from utils import AverageMeter
from dataset.loader import UPE, UPE_INF


parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='sum of epochs')
parser.add_argument('--eval_freq', default=10, type=int, help='validate frequency')
args = parser.parse_args()


class fixed_loss(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, out_image, gt_image, out_thumb, gt_thumb):
		ms_ssim_loss = 1 - ms_ssim(out_image, gt_image, data_range=1, size_average=True)
		l1_loss = F.l1_loss(out_thumb, gt_thumb)

		lambda1 = 1
		lambda2 = 1
		loss = lambda1 * l1_loss + lambda2 * ms_ssim_loss

		return loss


def train(train_loader, model, criterion, optimizer, epoch):
	losses = AverageMeter()
	model.train()

	for ind, (low_img, high_img, thumb_low_img, thumb_high_img, position) in enumerate(train_loader):
		st = time.time()

		input_var = low_img.cuda()
		target_var = high_img.cuda()
		thumb_input_var = thumb_low_img.cuda()
		thumb_target_var = thumb_high_img.cuda()
		position_var = position

		output, thumb_output = model(input_var, thumb_input_var, position_var)

		loss = criterion(output, target_var, thumb_output, thumb_target_var)
		losses.update(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print('[{0}][{1}]\t'
			'lr: {lr:.5f}\t'
			'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
			'Time: {time:.3f}'.format(
			epoch, ind,
			lr=optimizer.param_groups[-1]['lr'],
			loss=losses,
			time=time.time()-st))


def validate(val_loader, model, epoch):
	losses = AverageMeter()
	
	torch.cuda.empty_cache()
	model.eval()

	for ind, (low_img, high_img, thumb_low_img) in enumerate(val_loader):
		input_var = low_img.cuda()
		target_var = high_img.cuda()
		thumb_input_var = thumb_low_img.cuda()

		with torch.no_grad():
			output, _ = model(input_var, thumb_input_var)

		loss = F.l1_loss(output, target_var)
		losses.update(loss.item())

		print('Validation: [{0}][{1}]\t'
			'L1 Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
			epoch, ind,
			loss=losses))

	return losses.avg		


if __name__ == '__main__':
	save_dir = './save_model/'
	data_dir = './data/'

	model = ME()
	print(model)
	model.cuda()

	model = nn.DataParallel(model)

	if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
		print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']
		best_loss = model_info['loss']
	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		cur_epoch = 0
		best_loss = 1.0

	criterion = fixed_loss()
	criterion.cuda()

	train_dataset = UPE(os.path.join(data_dir, 'train'))
	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

	val_dataset = UPE_INF(os.path.join(data_dir, 'test'))
	val_loader = DataLoader(
		val_dataset, batch_size=1, shuffle=False, num_workers=8)

	for epoch in range(cur_epoch, args.epochs + 1):
		train(train_loader, model, criterion, optimizer, epoch)

		if epoch % args.eval_freq == 0:
			avg_loss = validate(val_loader, model, epoch)

			if avg_loss < best_loss:
				best_loss = avg_loss
				torch.save({
					'epoch': epoch + 1,
					'loss': best_loss,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict()}, 
					os.path.join(save_dir, 'best_model.pth.tar'))

		torch.save({
			'epoch': epoch + 1,
			'loss': best_loss,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()}, 
			os.path.join(save_dir, 'checkpoint.pth.tar'))

		print('Best validation L1 loss: %.4f' % best_loss)
		