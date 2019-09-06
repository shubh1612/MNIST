import load_data as ld
import kNbr as knn

path = 'F:\\Study\\Projects\\MNIST\\Data\\'
X_train, Y_train = ld.load_mnist(path+'train\\', 'train')
X_test, Y_test = ld.load_mnist(path+'test\\', 'test')

print(len(X_train), len(Y_train))
#### K Nearest Neightbours

## K Nearest Neighbours with L2 Norm and no deskewing
model, Y_pred = knn.simpleEuclideanL2(X_train, Y_train, X_test)
