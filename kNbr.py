from sklearn.neighbors import KNeighborsClassifier

def simpleEuclideanL2(X_train, Y_train = None, X_test = None, model = None):

	if (Y_train is None):
		Y_test = model.predict(X_test)
		return model, Y_test
	elif (X_test is None):
		model = KNeighborsClassifier(p = 2)
		model.fit(X_train, Y_train)
		return model, []
	else:
		model = KNeighborsClassifier(p = 2)
		model.fit(X_train, Y_train)
		Y_test = model.predict(X_test)
		return model, Y_test
