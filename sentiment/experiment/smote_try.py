"""
Module untuk melakukan oversampling
dengan menggunakan algoritma SMOTE
Synthetic Minority Oversampling Technique
"""

import random
import numpy as np
from sklearn.neighbors import NearestNeighbors



def SMOTE(X, y, N, k=5, random_seed=0):
	"""
	fungsi untuk melakukan oversampling
	dengan menggunakan algoritma SMOTE (Synthetic Minority Oversampling Technique)
	
	return X_res, y_res: data (X, y) hasil oversampling
	
	parameter:
	X = data yang akan di-oversampling
	y = label data, terbatas untuk klasifikasi biner, dengan nilai 0 dan 1
	N = berapa persen oversampling yang akan dilakukan, asumsi kelipatan 100 persen
	k = k neirest neighbor yang akan digunakan pada smote (default=5)
	random_seed = seed untuk generate random
	"""

	X, y = np.array(X), np.array(y)

	if len(X) != len(y):
		raise Exception(
			'X and Y not in the same length. X: {}, Y: {}'
			.format(len(X), len(y)))

	# tidak dilakukan oversampling
	if N < 100:
		return X, y


	data_bundled = list(zip(X, y))
	X_kelas_pertama = np.array([x for x, y in data_bundled if y == 1])
	X_kelas_kedua = np.array([x for x, y in data_bundled if y == 0])

	n_kelas_pertama = len(X_kelas_pertama)
	n_kelas_kedua = len(X_kelas_kedua)

	print('n_kelas_pertama', n_kelas_pertama)
	print('n_kelas_kedua', n_kelas_kedua)

	if n_kelas_pertama < n_kelas_kedua:
		label_minor = 1
		X_minor = X_kelas_pertama
		n_minor = n_kelas_pertama
		X_mayor = X_kelas_kedua
		n_mayor = n_kelas_kedua
	else:
		label_minor = 0
		X_minor = X_kelas_kedua
		n_minor = n_kelas_kedua
		X_mayor = X_kelas_pertama
		n_mayor = n_kelas_pertama


	"""
	mendefinisikan beberapa variabel

	variabel:
	N = berapa kali dari banyak n data minor yang akan dihasilkan
	n_attrs = banyak atribut/dimensi dari data
	n_generated = counter berapa banyak data yang sudah d-generate
	X_generated = X data hasil oversampling
	"""
	N = int(N/100)
	n_attrs = len(X[0])
	n_generated = 0
	X_generated = np.zeros(shape=(N*n_minor, n_attrs))

	print('N', N)
	print('n_attrs', n_attrs)


	# get nearest neighbors dari masing-masing data X kelas minor
	# kemudian simpan k index-indexnya
	neighbors = NearestNeighbors(n_neighbors=k).fit(X_minor) \
		.kneighbors(n_neighbors=k, return_distance=False)

	print('neighbors:\n', neighbors)
	print()


	# proses generasi data sintetis
	for i in range(n_minor):
		nn_array = neighbors[i]
		_N = N
		nn_indices = np.arange(0, k).tolist()

		print('nn_array', nn_array)
		print('_N', _N)

		for j in range(N):
			# random.seed(random_seed+i+N)
			nn_indices_chosen_N = random.choice(nn_indices)
			nn_indices = [i for i in nn_indices if i != nn_indices_chosen_N]
			nn_index = nn_indices_chosen_N

			print('\t nn_indices_chosen_N', nn_indices_chosen_N)
			print('\t nn_indices', nn_indices)
			print('\t nn_index', nn_index)

			for attr in range(n_attrs):
				distance = X_minor[nn_array[nn_index]][attr] - X_minor[i][attr]
				random.seed(random_seed+i+N+attr)
				gap = random.random()
				X_generated[n_generated][attr] = X_minor[i][attr] + gap*distance

				print('\t\t # distance', distance)
				print('\t\t gap', gap)
				print('\t\t X_minor[i][attr]: {}, gap*distance: {}'.format(X_minor[i][attr], gap*distance))
				print('\t\t X_generated[n_generated][attr]', X_generated[n_generated][attr])
				print()
			
			n_generated += 1
			_N -= 1

		print()

	X_res = np.concatenate(
		[X_mayor, X_minor, X_generated], axis=0)
	y_res = np.concatenate(
		[y, [label_minor for i in range(n_generated)]], axis=0)
	

	# mengembalikan data hasil oversampling
	return X_res, y_res



if __name__ == '__main__':
	X = [
		[1,1,1,1,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,0,1,1,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0],
		[1,1,1,0,0,0,0,0],
		[1,0,1,1,0,0,0,0],
		[1,0,1,0,0,0,0,0],
		[1,1,0,1,0,0,0,0],
		[0,1,0,1,0,0,0,0],
		[1,0,0,1,0,0,0,0],

		[0,0,0,0,1,1,1,1],
		[0,0,0,0,0,1,1,1],
		[0,0,0,0,0,0,1,1],
		[0,0,0,0,0,0,0,1],
	]

	y = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]

	X_res, y_res = SMOTE(X, y, 200, k=3)
	print(X_res, y_res)
	# print(y_res)