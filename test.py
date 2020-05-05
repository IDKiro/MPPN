from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil, importlib, argparse
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from skimage.measure import compare_psnr, compare_ssim
from torch.utils.data import DataLoader

from model.ME import Network as ME
from utils import AverageMeter, chw_to_hwc
from dataset.loader import UPE_INF


def test(test_loader, model, result_dir):
	psnr = AverageMeter()
	ssim = AverageMeter()

	model.eval()

	for ind, (low_img, high_img, thumb_low_img) in enumerate(test_loader):
		input_var = low_img.cuda()
		thumb_input_var = thumb_low_img.cuda()

		with torch.no_grad():
			output, _ = model(input_var, thumb_input_var)

		low_np = low_img.numpy()
		high_np = high_img.numpy()
		output_np = output.cpu().detach().numpy()

		low_np_img = chw_to_hwc(low_np[0])
		high_np_img = chw_to_hwc(high_np[0])
		output_img = chw_to_hwc(np.clip(output_np[0], 0, 1))

		test_psnr = compare_psnr(high_np_img, output_img, data_range=1)
		test_ssim = compare_ssim(high_np_img, output_img, data_range=1, multichannel=True)

		psnr.update(test_psnr)
		ssim.update(test_ssim)

		print('Testing: [{0}]\t'
			'PSNR: {psnr.val:.2f} ({psnr.avg:.2f})\t'
			'SSIM: {ssim.val:.3f} ({ssim.avg:.3f})'.format(
			ind,
			psnr=psnr,
			ssim=ssim))

		if not os.path.isdir(result_dir):
			os.makedirs(result_dir)

		temp = np.concatenate((low_np_img, high_np_img, output_img), axis=1)
		io.imsave(os.path.join(result_dir, 'test_%d.jpg'%ind), np.uint8(temp * 255))


if __name__ == '__main__':
	save_dir = './save_model/'
	result_dir = './result/'

	model = ME(conv3_num=0, conv1_num=6)
	print(model)
	model.cuda()

	model = nn.DataParallel(model)

	if os.path.exists(os.path.join(save_dir, 'best_model.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'best_model.pth.tar'))
		print('==> loading existing model:', os.path.join(save_dir, 'best_model.pth.tar'))
		model.load_state_dict(model_info['state_dict'])
	else:
		exit(1)

	test_dataset = UPE_INF('../dataset/UPE/test_full/')
	test_loader = DataLoader(
		test_dataset, batch_size=1, shuffle=False)

	test(test_loader, model, result_dir)