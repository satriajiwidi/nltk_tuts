from collections import Counter
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE 


# X, y = make_classification(
# 	n_classes=2, weights=[0.4, 0.6],
# 	n_features=5, n_samples=20, random_state=10
# )

X = [
	[1, 1, 1, 0, 0, 0],
	[1, 0, 1, 0, 0, 0],
	[1, 1, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 0],
	[0, 1, 1, 0, 0, 0],
	[0, 0, 1, 0, 0, 0],

	[0, 0, 0, 1, 1, 1],
	[0, 0, 0, 1, 1, 0],
	[0, 0, 0, 1, 0, 0],
	[0, 0, 0, 1, 1, 0],
]

y = [1,1,1,1,1,1,1,0,0,0,0]


sm = SMOTE(random_state=40, k_neighbors=3)
X_res, y_res = sm.fit_sample(X, y)
resampled_data = [x for x, y in zip(X_res, y_res) if y==0]

from pprint import pprint
# pprint(resampled_data)
# print('Original dataset shape {}'.format(Counter(y)))
# print('Resampled dataset shape {}'.format(Counter(y_res)))


from sklearn.model_selection import train_test_split
nb = MultinomialNB()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42)
nb.fit(X_train, y_train)
# print(nb.score(X_test, y_test))

print(X_res)
print(y_res)