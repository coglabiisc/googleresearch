import os
import sys
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt

import utils


def main(args):
	np.random.seed(args.seed)

	shifts = ['aptos']
	models = [
		'deterministic_k1_indomain_mc5/single',
		'dropout_k1_indomain_mc5/single',
		'vi_k1_indomain_mc5/single',
	]
	datasets = ['in_domain_test', 'ood_test']

	results = utils.read_arrays(args.results_path)


	val_split_dict = {}


	aupcc_auroc = lambda outputs: np.mean(outputs['auroc'])
	aupcc_accuracy = lambda outputs: np.mean(outputs['accuracy'])
	aupcc_avgprec = lambda outputs: np.mean(outputs['avgprec'])
	aupcc_f1 = lambda outputs: np.mean(outputs['f1'])

	colors = {
		'base': 'k',
		'SR': 'r',
		'SSR': 'b',
		'SPS': 'y',
	}

	for shift, model, dataset in itertools.product(shifts, models, datasets):

		# load arrays
		model_tuning, ensemble = model.split('/')
		arr_list = results[shift][model_tuning][ensemble][dataset].values()

		# output paths
		path = os.path.join(args.plots_path, shift, mt, ensemble, dataset)
		os.makedirs(path, exist_ok = True)


		# get validation split
		val_split = val_split_dict.get((shift, dataset))
		if val_split is None:
			val_split = val_split_dict[(shift, dataset)] = utils.get_val_split(arr_list, frac = args.validation_fraction)

		val_arr_list = utils.split_arr_list(arr_list, val_split)
		test_arr_list = utils.split_arr_list(arr_list, ~val_split)


		base = utils.original_referral_curves(test_arr_list)
		
		SR = utils.split_referral_curves(test_arr_list)

		SPS_params, SSR_params = {}, {}

		if dataset == 'ood_test':
			SPS_params['aupcc_auroc'] = utils.optimize_split_plattscale(
				val_arr_list, aupcc_auroc)
			SPS_params['aupcc_accuracy'] = utils.optimize_split_plattscale(
				val_arr_list, aupcc_accuracy)
			SSR_params['aupcc_auroc'] = utils.optimize_shifted_split_referral(
				val_arr_list, aupcc_auroc)
			SSR_params['aupcc_accuracy'] = utils.optimize_shifted_split_referral(
				val_arr_list, aupcc_accuracy)

		if dataset == 'in_domain_test':
			mm_metric = utils.get_mean_matching_metric(base)
		elif dataset == 'ood_test':
			SPS_params['mean_match'] = utils.optimize_constrained_split_plattscale(
				val_arr_list, mm_metric)
			SSR_params['mean_match'] = utils.optimize_shifted_split_referral(
				val_arr_list, mm_metric)

		SPS = {
			metric: utils.split_plattscaled_referral_curves(test_arr_list, *params)
			for metric, params in SPS_params.items()
		}
		SSR = {
			metric: utils.shifted_split_referral_curves(test_arr_list, params)
			for metric, params in SSR_params.items()
		}


		# plot
		for pfm in ('auroc', 'accuracy'):
			plt.figure(figsize = (9, 9))
			plt.suptitle(f'{shift} -- {dataset}\n{obj} -- {mt} -- {ensemble}')

			plot_curve(
				base[pfm],
				base[f'{pfm}_stddev'],
				color = colors['base'])
			plot_curve(
				SR[pfm],
				SR[f'{pfm}_stddev'],
				color = colors['SR'])

			if dataset == 'in_domain_test':
				plot_curve(
					SSR[f'aupcc_{pfm}'][pfm],
					SSR[f'aupcc_{pfm}'][f'{pfm}_stddev'],
					color = colors['SSR'])
				plot_curve(
					SPS[f'aupcc_{pfm}'][pfm],
					SPS[f'aupcc_{pfm}'][f'{pfm}_stddev'],
					color = colors['SPS'])

			elif dataset == 'ood_test':
				plot_curve(
					SPS[f'aupcc_{pfm}'][pfm],
					SPS[f'aupcc_{pfm}'][f'{pfm}_stddev'],
					color = colors['SPS'], linestyle = '--')
				plot_curve(
					SSR[f'aupcc_{pfm}'][pfm],
					SSR[f'aupcc_{pfm}'][f'{pfm}_stddev'],
					color = colors['SSR'], linestyle = '--')
				plot_curve(
					SPS['mean_match'][pfm],
					SPS['mean_match'][f'{pfm}_stddev'],
					color = colors['SPS'])
				plot_curve(
					SSR['mean_match'][pfm],
					SSR['mean_match'][f'{pfm}_stddev'],
					color = colors['SSR'])


			plt.xlabel('Referral Rate', fontsize=36)
			plt.ylabel(pfm.upper(), fontsize=36)

			plt.xticks(fontsize = 24)
			plt.yticks(fontsize = 24)

			plt.tight_layout()
			plt.savefig(os.path.join(path, f'{pfm}-ref.png'))

			plt.close()


		# print aupcc
		for pfm in ('auroc', 'accuracy'):
				print('=' * 79)
				print(f'{pfm} @ {model}')
				print('-' * 79)
				print(f'SPS (aupcc-{pfm}):', SPS[f'aupcc_{pfm}'][pfm].mean())
				print(f'SSR (aupcc-{pfm}):', SSR[f'aupcc_{pfm}'][pfm].mean())
				if dataset == 'ood_test':
					print(f'SPS (mean-match):', SPS['mean_match'][pfm].mean())
					print(f'SSR (mean-match):', SSR['mean_match'][pfm].mean())


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--seed', type = int, default = 0, help = 'numpy random seed')
	parser.add_argument('--validation_fraction', type = float, default = 0.5, help = 'fraction of test set to hold out for tuning parameters')
	parser.add_argument('--results_path', type = str, help = 'path to directory containing evaluation results')
	parser.add_argument('--plots_path', type = str, help = 'path to output directory for storing plots')

	args = parser.parse_args()
	main(args)

