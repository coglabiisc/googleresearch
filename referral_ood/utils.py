import os
import copy

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def ecdf(arr):
	return (arr.argsort().argsort() + 1) / (arr.size + 1)


def compute_metric(metric_fn, labels, outputs, default = 0):
	try:
		return metric_fn(labels, outputs)
	except ValueError:
		return default


def get_referral_curves(arr_list, dec_th, max_referral_rate = 95, from_logit = True):
	auroc_list, accuracy_list = [], []
	f1_list, avgprec_list = [], []

	all_logits = []

	for arr in arr_list:
		labels = arr['y_true']
		
		if from_logit:
			outputs = arr['y_logit']
			order = arr.get('referral_order', np.argsort(-np.abs(outputs)))
			all_logits.append(outputs)
		else:
			outputs = arr['y_pred']
			order = arr['y_pred_entropy'].argsort()

		ret_auroc, ret_accuracy = [], []
		ret_f1, ret_avgprec = [], []

		for r in range(max_referral_rate):
			ids = order[:int((100 - r) * order.size / 100)]

			ret_accuracy.append(compute_metric(
				metrics.accuracy_score,
				labels[ids], outputs[ids] >= dec_th
			))
			ret_f1.append(compute_metric(
				metrics.f1_score,
				labels[ids], outputs[ids] >= dec_th
			))
			ret_avgprec.append(compute_metric(
				metrics.average_precision_score,
				labels[ids], outputs[ids]
			))
			ret_auroc.append(compute_metric(
				metrics.roc_auc_score,
				labels[ids], outputs[ids]
			))

		auroc_list.append(ret_auroc)
		accuracy_list.append(ret_accuracy)
		f1_list.append(ret_f1)
		avgprec_list.append(ret_avgprec)

	aurocs = np.mean(auroc_list, axis = 0)
	accuracies = np.mean(accuracy_list, axis = 0)
	aurocs_stddev = np.std(auroc_list, axis = 0)
	accuracies_stddev = np.std(accuracy_list, axis = 0)
	f1s = np.mean(f1_list, axis = 0)
	avgprecs = np.mean(avgprec_list, axis = 0)
	f1s_stddev = np.std(f1_list, axis = 0)
	avgprecs_stddev = np.std(avgprec_list, axis = 0)

	all_logits = np.array(all_logits)

	return dict(
		auroc = aurocs,
		accuracy = accuracies,
		f1 = f1s,
		avgprec = avgprecs,
		all_logits = all_logits,
		auroc_stddev = aurocs_stddev,
		accuracy_stddev = accuracies_stddev,
		f1_stddev = f1s_stddev,
		avgprec_stddev = avgprecs_stddev,
	)


def original_referral_curves(arr_list, dec_th = 0):
	return get_referral_curves(arr_list, dec_th)


def split_referral_curves(arr_list, dec_th = 0):
	arr_list = copy.deepcopy(arr_list)

	for arr in arr_list:
		logits = arr['y_logit']

		pos = logits >= dec_th
		neg = logits < dec_th

		cdf = np.zeros_like(logits)
		cdf[pos] = ecdf(logits[pos])
		cdf[neg] = ecdf(logits[neg]) - 1
		arr['referral_order'] = np.argsort(-np.abs(cdf))

	return get_referral_curves(arr_list, dec_th)


def shifted_split_referral_curves(arr_list, b_list, dec_th = 0):
	arr_list = copy.deepcopy(arr_list)

	for arr, b in zip(arr_list, b_list):
		logits = arr['y_logit']
		logits[...] += b

		pos = logits >= dec_th
		neg = logits < dec_th

		cdf = np.zeros_like(logits)
		cdf[pos] = ecdf(logits[pos])
		cdf[neg] = ecdf(logits[neg]) - 1
		arr['referral_order'] = np.argsort(-np.abs(cdf))

	return get_referral_curves(arr_list, dec_th)


def optimize_shifted_split_referral(arr_list, objective, dec_th = 0):

	def reg(b):
		return 0

	best_b_list = []

	if hasattr(objective, '__iter__'):
		objective_list = objective
	else:
		objective_list = [objective] * len(arr_list)

	for arr, obj in zip(arr_list, objective_list):
		depth, half_width, stepsize = 7, 2, 32
		obj_val = {0: obj(shifted_split_referral_curves(
							[arr],
							[0],
							dec_th = dec_th
						)) - reg(0)}

		for _ in range(depth):
			for b in max(obj_val, key = obj_val.get) + stepsize * np.arange(-half_width, half_width+1):
				if b not in obj_val:
					obj_val[b] = obj(
						shifted_split_referral_curves(
							[arr],
							[b],
							dec_th = dec_th
						)) - reg(b)

			stepsize /= half_width

		best_b_list.append(max(obj_val, key = obj_val.get))

	return best_b_list


def split_plattscaled_referral_curves(arr_list, ar_list, b_list, dec_th = 0):
	arr_list = copy.deepcopy(arr_list)

	for arr, ar, b in zip(arr_list, ar_list, b_list):
		logits = arr['y_logit']
		logits[...] += b
		if ar > 1:
			logits[logits < dec_th] = logits[logits < dec_th] / ar
		else:
			logits[logits >= dec_th] = logits[logits >= dec_th] * ar

	return get_referral_curves(arr_list, dec_th)


def optimize_split_plattscale(arr_list, objective, dec_th = 0):

	def reg(lar, b):
		return 0

	best_lar_list, best_b_list = [], []

	if hasattr(objective, '__iter__'):
		objective_list = objective
	else:
		objective_list = [objective] * len(arr_list)

	for arr, obj in zip(arr_list, objective_list):
		depth = 7
		hw_lar, s_lar = 2, 3
		hw_b, s_b = 2, 32

		obj_val = {(0, 0): obj(split_plattscaled_referral_curves(
								[arr],
								[1], [0],
								dec_th = dec_th
							)) - reg(0, 0)}

		for _ in range(depth):
			best_lar, best_b = max(obj_val, key = obj_val.get)

			for lar in best_lar + s_lar * np.arange(- hw_lar, hw_lar + 1):
				for b in best_b + s_b * np.arange(- hw_b, hw_b + 1):
					if (lar, b) not in obj_val:
						obj_val[(lar, b)] = obj(
							split_plattscaled_referral_curves(
								[arr],
								[np.exp(lar)], [b],
								dec_th = dec_th
							)) - reg(lar, b)

			s_lar /= hw_lar
			s_b /= hw_b

		best_lar, best_b = max(obj_val, key = obj_val.get)
		best_lar_list.append(best_lar)
		best_b_list.append(best_b)

	return np.exp(best_lar_list), best_b_list


def optimize_constrained_split_plattscale(arr_list, objective, dec_th = 0):

	def reg(ar, b):
		return 1e-8 * b ** 2

	best_ar_list, best_b_list = [], []

	if hasattr(objective, '__iter__'):
		objective_list = objective
	else:
		objective_list = [objective] * len(arr_list)

	for arr, obj in zip(arr_list, objective_list):
		depth = 7
		hw_b, s_b = 2, 32

		obj_val = {(1, 0): obj(split_plattscaled_referral_curves(
								[arr],
								[1], [0],
								dec_th = dec_th
							)) - reg(1, 0)}
		ar_from_b = {}

		for _ in range(depth):
			best_ar, best_b = max(obj_val, key = obj_val.get)

			for b in best_b + s_b * np.arange(- hw_b, hw_b + 1):
				if b not in ar_from_b:
					ll = split_plattscaled_referral_curves(
								[arr],
								[1], [b],
								dec_th = dec_th
							)['all_logits']
					ar_from_b[b] = np.abs(np.min(ll) / np.max(ll))

				ar = ar_from_b[b]

				if (ar, b) not in obj_val:
					obj_val[(ar, b)] = obj(
						split_plattscaled_referral_curves(
							[arr],
							[ar], [b],
							dec_th = dec_th
						)) - reg(ar, b)

			s_b /= hw_b

		best_ar, best_b = max(obj_val, key = obj_val.get)
		best_ar_list.append(best_ar)
		best_b_list.append(best_b)

	return best_ar_list, best_b_list


def read_arrays(path):
	if os.path.isdir(path):
		d = {}
		for child in os.listdir(path):
			child_name, _ = os.path.splitext(child)
			child_path = os.path.join(path, child)
			d[child_name] = read_arrays(child_path)
		return d
	elif os.path.splitext(path)[1] == '.npy':
		return np.load(path, allow_pickle = True)
	else:
		return None


def get_val_split(arr_list, frac = 0.5):
	N = arr_list[0]['y_true'].size
	val_split = np.zeros(N, bool)
	
	ids = np.random.choice(N, round(N * frac), replace = False)
	val_split[ids] = True
	
	return val_split


def split_arr_list(arr_list, split):
	return [
		{k: v[split] for k, v in arr.items() if v.size == split.size}
		for arr in arr_list
	]


def plot_curve(arr, err = 0, **kwargs):
	kwargs.setdefault('linewidth', 4)
	plt.plot(np.arange(arr.size) / 100, arr, *args, **kwargs)
	kwargs['alpha'] = 0.05
	plt.fill_between(np.arange(arr.size) / 100, arr - err, arr + err, *args, **kwargs)


def get_mean_matching_metric(id_outputs, var_reg_coeff = 1e-2):
	id_logit_list = id_outputs['all_logits']
	neg = id_logit_list < 0
	pos = id_logit_list >= 0

	id_m0 = (id_logit_list * neg).sum(1) / (neg.sum(1) + 1e-16)
	id_m1 = (id_logit_list * pos).sum(1) / (pos.sum(1) + 1e-16)

	def metric_for_seed(seed_idx):
		def metric(outputs):
			ood_logit_list = outputs['all_logits'][0] # or flatten()?
			neg = ood_logit_list < 0
			pos = ood_logit_list >= 0

			ood_m0 = (ood_logit_list * neg).sum() / (neg.sum() + 1e-16)
			ood_m1 = (ood_logit_list * pos).sum() / (pos.sum() + 1e-16)
			
			dev = np.linalg.norm(id_m0[seed_idx] - ood_m0) ** 2 + np.linalg.norm(id_m1[seed_idx] - ood_m1) ** 2
			var = ood_logit_list[neg].var() + ood_logit_list[pos].var()

			return -(dev + var_reg_coeff * var)
		return metric
	
	return [metric_for_seed(idx) for idx in range(len(id_logit_list))]

