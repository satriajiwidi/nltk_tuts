"""
Module berisi beberapa fungsi untuk melakukan proses-
pembentukan model terbaik dari masing-masing classifier-
dan jenis fitur yang diberikan

1. get_best_model
2. do_training_testing
3. do_validation
"""

import csv, os
import numpy as np

from metrics import accuracy_score, \
	confusion_matrix, \
	geometric_mean_score as gmean

from kfold import kfold



def get_best_model(X, y, clf, kf, clf_name, fitur, filename, show=False):
	"""
	fungsi untuk mendapatkan model terbaik dari hasil k-fold

	return best_model: model terbaik, dengan tolak ukur gmean
	
	parameter:
	X = data per jenis fitur
	y = label dari data
	clf = object classifier
	kf = object K-Fold
	show = boolean untuk mencetak proses pencarian model terbaik
	"""

	performance_total = 0
	best_fold_performance = -100
	best_fold_index = -1
	best_model = None

	if show: print('\t\t', end='')

	performances = []
	performances.append(clf_name.upper() + '-' + fitur.upper())

	for index, (train_index, test_index) in enumerate(kf):
		X_train_fold, X_test_fold = X[train_index], X[test_index]
		y_train_fold, y_test_fold = y[train_index], y[test_index]
		clf_now = clf.fit(X_train_fold, y_train_fold)
		pred = clf.predict(X_test_fold)
		acc = round(accuracy_score(y_test_fold, pred) * 100, 2)
		gmean_score = round(gmean(y_test_fold, pred) * 100, 2)
		selected_metric_score = gmean_score

		if selected_metric_score > best_fold_performance:
			best_fold_performance = selected_metric_score
			best_fold_index = index
			best_model = clf_now

		if show: print(selected_metric_score, end=' ')

		performance_total += selected_metric_score

		performances.append(selected_metric_score)

	performance_avg = round(performance_total / 10, 2)
	performances.append(performance_avg)

	if show:
		print('\n\t\tbest index: {}, best performance: {}, performance avg: {}\n'
			.format(best_fold_index+1, best_fold_performance, performance_avg))

	with open(filename, 'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(performances)

	return best_model


def do_training_testing(clf, X, y, filename, show=False):
	"""
	fungsi untuk melakukan training dan testing
	baik itu dengan atau tanpa resampling
	return per_clf: model terbaik dari masing-masing fitur
	
	parameter:
	clf = array object classifier
	X = data per jenis fitur
	y = label dari data
	kf = object K-Fold
	show = boolean, untuk mencetak proses pencarian model terbaik
	"""

	try:
		os.remove(filename)

	except OSError:
		pass

	first_row = ['Clf-Fitur']
	for i in range(10):
		first_row.append('Fold ' + str(i+1))
	first_row.append('Avg')
	with open(filename, 'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(first_row)

	per_clf = {}
	
	train_indices_all, test_indices_all = kfold(y, n_splits=10)
	kf = np.array(list(zip(train_indices_all, test_indices_all)))

	for c in clf: # untuk masing-masing jenis classifier
		if c == 'gauss_nb':
			continue
		
		for index, fitur in enumerate(X): # untuk masing-masing jenis fitur
			y_train = y

			c1 = False
			if c == 'multi_nb' and fitur == 'tfidf':
				c1 = True
				c = 'gauss_nb'

			
			if show: # show process
				print('\t', c, fitur)
				per_clf[(c, fitur)] = get_best_model(
					X[fitur], y_train, clf[c], kf, c, fitur, filename, show=True)
			else:
				per_clf[(c, fitur)] = get_best_model(
					X[fitur], y_train, clf[c], kf, c, fitur, filename)

			if c1:
				c = 'multi_nb'

	return per_clf



def do_validation(clf, X_val, y_val, per_clf, filename):
	"""
	fungsi untuk melakukan validasi
	baik itu dengan atau tanpa resampling
	
	parameter:
	clf = array object classifier
	X_val = data per jenis fitur khusus untuk proses validasi
	y_val = label dari data khusus untuk proses validasi
	per_clf = model terbaik untuk masing-masing fitur
	"""

	try:
		os.remove(filename)

	except OSError:
		pass

	first_row = ['Clf-Fitur', 'Skor']
	with open(filename, 'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(first_row)

	for c in clf: # untuk masing-masing jenis classifier
		if c == 'gauss_nb':
			continue

		performance_total = 0
		n_fitur = 0

		for index, fitur in enumerate(X_val):
			n_fitur += 1

			c1 = False
			if c == 'multi_nb' and fitur == 'tfidf':
				c1 = True
				c = 'gauss_nb'
			
			pred = per_clf[c, fitur].predict(X_val[fitur])
			
			performances = []
			# performa menggunakan akurasi atau gmean
			
			gmean_score = round(gmean(y_val, pred) * 100, 2)
			acc = round(accuracy_score(y_val, pred) * 100, 2)
			# print(confusion_matrix(y_val, pred))
			performance_total += gmean_score

			if c == 'multi_nb' or c == 'gauss_nb':
				print('{} {}: \t\t {}'.format(c, fitur, gmean_score))
			else:
				print('{} {}: \t\t\t {}'.format(c, fitur, gmean_score))
			performances.append(c.upper() + '-' + fitur.upper())

			performances.append(gmean_score)

			with open(filename, 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(performances)

			if c1:
				c = 'multi_nb'

		performance_avg = round(performance_total / n_fitur, 2)

		print('>> performa rata2 {}: \t({})'.format(c, performance_avg))
