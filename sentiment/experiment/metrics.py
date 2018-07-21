import numpy as np
import math

def confusion_matrix(y_true, y_pred, table_show=True):
	FIRST_CLASS = 1
	SECOND_CLASS = 0

	zipped = np.array(list(zip(y_true, y_pred)))
	tp, fn, fp, tn = 0, 0, 0, 0

	for y_true, y_pred in zipped:
		if y_true == y_pred and y_true == FIRST_CLASS:
			tp += 1
		elif y_true == y_pred and y_true == SECOND_CLASS:
			tn += 1
		elif y_true != y_pred and y_true == SECOND_CLASS:
			fp += 1
		else:
			fn += 1

	if table_show:
		return np.array([tp, fn, fp, tn]).reshape([2,2])

	return tp, fn, fp, tn


def geometric_mean_score(y_true, y_pred):
	tp, fn, fp, tn = confusion_matrix(y_true, y_pred, table_show=False)

	return math.sqrt((tn / (tn+fp)) * (tp / (tp+fn)))


def accuracy_score(y_true, y_pred):
	tp, fn, fp, tn = confusion_matrix(y_true, y_pred, table_show=False)

	return (tp+tn) / (tp+tn+fn+fp)


if __name__ == '__main__':
	Y_true = [1,1,1,1,1,0,0,0,0,0]
	Y_pred = [1,0,1,1,1,0,1,0,0,1]

	cm = confusion_matrix(Y_true, Y_pred)
	gm = geometric_mean_score(Y_true, Y_pred)
	acc = accuracy_score(Y_true, Y_pred)
	print(cm)
	print(gm)
	print(acc)