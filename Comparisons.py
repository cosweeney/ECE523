#Imports_________________________________________________________________________________________________________
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow import keras
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from tqdm import tqdm #for progress bars
from scipy.stats import multivariate_normal as mvnorm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib
matplotlib.rcParams.update({'text.usetex': True, 
                            'font.family': 'Computer Modern Roman'})

import warnings
warnings.filterwarnings('ignore')


#Load in data, Normalize and format
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

#Normalize and format
train_images = train_images / 255 
test_images  = test_images / 255


#Reshape data: flatten individual 32x32x3 vectors to 1d vectors of length 32*32*3=3072  
train_img = np.reshape(train_images, (train_images.shape[0], -1)).astype("float")
test_img = np.reshape(test_images, (test_images.shape[0], -1)).astype("float")


#Convolutional Neural Network___________________________________________________________________________________

#Load saved network, which was desined & trained in CIFAR10_CNN.ipynb
model = keras.models.load_model('myCNN.h5')
model.compile(loss ='categorical_crossentropy', optimizer ='adam', metrics =['accuracy'])


#Bayes Classifier________________________________________________________________________________________________

#Define Bayes Classifier:

class Bayes_Classifier(object):
    def __init__(self):
        pass
    def fit(self, x, y):
        """
        Load in training data and compute 
        means, covariances, and priors for 
        each class. 
        __________________________________
        x: training data
        y: training labels

        returns: means, covariances, and priors.
        """
        means  = []
        covs   = []
        pri    = []
        
        label_list = np.unique(y)
        for i in range(len(label_list)):
            #Mask for current class
            mask = (y[:, 0] == label_list[i])
            xm = x[mask]

            #Append class mean, cov, prior prob
            means.append( np.mean(xm, axis=0) )
            covs.append( np.cov(xm.T) )
            pri.append( xm.shape[0]/x.shape[0] )
        

        self.means = np.array(means)
        self.covs  = np.array(covs)
        self.pri   = np.array (pri)
        return means, covs, pri


    def predict(self, x):
        """ 
        Predict labels of test data using posterior probabilities.
        __________________________________________________________
        x: training data

        returns: class of data. 
        """
        label_num = self.means.shape[0]

        # Calculate posterior probabilities
        probs = np.array([self.pri[i]*mvnorm.pdf(x, mean=self.means[i], cov=self.covs[i]) for i in range(label_num)])

        total = np.sum(probs, axis=0)

        norm_probs = probs / total

        #Assign class of maximum probability 
        mclass = np.argmax(norm_probs, axis=0)

        return np.array(mclass)
        
    def evaluate(self, x, y):
        """ 
        Evaluate accuracy of classification. 
        ____________________________________
        x: test data
        y: test labels
        """
        preds = self.predict(x)
        score = accuracy_score(y, preds)

        n_corr = np.sum(preds == y)
        n_tot  = x.shape[0]
        print('BC Accuracy: {:.2f}%'.format(100*score))
        return score, n_corr, n_tot


#K-Nearest Neighbor______________________________________________________________________________________________

#Define the K Nearest Neighbor classifer:
class KNN(object):
    def __init__(self):
        pass
    def fit(self, x, y):
        """
        Load in training data. 
        ______________________
        x: training data
        y: training labels
        """
        self.traindata = x
        self.trainlabels = y
    
    def distances(self, x):
        """ 
        Compute Euclidean distances between training and test data 
        in feature space. 
        Makes use of broadcasting for efficiency, motivated by 
        https://ryli.design/blog/knn
        and 
        https://cs231n.github.io/classification/.
        ___________________________________________________________
        x: test data

        returns: d, euclidean distances between each training and 
                 test vector ( shape ( x.shape[0], self.traindata.shape[0] ) )
        """
        d = np.sqrt(np.sum(np.square(self.traindata), axis=1) +
                    np.sum(np.square(x), axis=1)[:, np.newaxis] -
                    2 * np.dot(x, self.traindata.T))
        return d
    
    def predict(self, dists, k=5):
        """ 
        Predict labels of input testing data. 
        _____________________________________
        dists: Euclidean distances between 
               test and train data. 
        k:     Integer number of nearest neighbors
               to consider for label prediction. 

        returns: label predictions for test data. 
        """
        nearest_labels=[]
        print('Predicting labels...')
        for i in tqdm(range(dists.shape[0])):

            #sort and collect nearest K distances
            cd = np.argsort(dists[i])[0:k] 

            #select most common labels from K Nearest neighbors
            nearest_label = self.trainlabels[ cd ]
            nearest_labels.append(np.argmax(np.bincount(nearest_label[:, 0])))
        
        nearest_labels=np.array(nearest_labels)
        return nearest_labels
    
    def evaluate(self, x, y, k=5):
        """ 
        Evaluate accuracy of classification for a value of k. 
        _____________________________________________________
        x: test data
        y: test labels
        k: Integer number of nearest neighbors to consider for
           label prediction. 
        """
        dists = self.distances(x)
        preds = self.predict(dists, k=k)
        score = accuracy_score(y, preds)

        n_corr = np.sum(preds == y)
        n_tot  = x.shape[0]
        print('Accuracy: {:.3f}%'.format(100*score))
        return score, n_corr, n_tot
    

#Comparison: Accuracy & Confusion Matrices
#CNN_____________________________________________________________________________________________________________
cpreds = model.predict(test_images)

print("CNN Accuracy: {:.2f}%".format(100*accuracy_score(test_labels, np.argmax(cpreds, axis=1) )) )

pa = np.zeros(test_labels[0].shape)
mpreds = np.zeros_like(cpreds)
rows = np.arange(len(cpreds))
mpreds[rows, np.argmax(cpreds, axis=1)] = 1

cm = confusion_matrix(test_labels, np.argmax(cpreds, axis=1))

CM = ConfusionMatrixDisplay(cm)
CM.plot().figure_.savefig('plots/CNN_CM.png')
plt.title('CNN Confusion')

#BC______________________________________________________________________________________________________________

#Compute PCA Features
X = train_images.reshape(-1, 3072) #flatten images: reshape data to Nx(32*32*3)

R = np.dot(X.T, X)                 #Compute autocorrelation matrix of data

evals, evecs = np.linalg.eig(R) #eigendecomposition of R

#Compute number of eigen-images needed to capture 99% of data covariance
cutoff = 0.99
cov_sum = np.cumsum(evals)/np.sum(evals)
k = np.argmax(cov_sum>=cutoff)

#Apply dimensionality reduction
X_red = np.dot(X, evecs[:,:k])
Xt = test_images.reshape(-1, 3072)
Xt_red = np.dot(test_images.reshape(-1, 3072), evecs[:,:k])


#Plot and Save Confusion Matrices

BC = Bayes_Classifier()
stats=BC.fit(X_red, train_labels)
preds = BC.predict(Xt_red)
print('Bayes Classifier with PCA:')
BC.evaluate(Xt_red, test_labels)


cm = confusion_matrix(test_labels, preds)

CM = ConfusionMatrixDisplay(cm)
CM.plot().figure_.savefig('plots/BC_CM.png')
plt.title('BC Confusion')

#KNN_____________________________________________________________________________________________________________

KC = KNN()
stats=KC.fit(train_img, train_labels)
kpreds = KC.predict(test_img, k=7)
KC.evaluate(test_img, test_labels, k=7)

cm = confusion_matrix(test_labels, kpreds)

CM = ConfusionMatrixDisplay(cm)
CM.plot().figure_.savefig('plots/KNN_CM.png')
plt.title('KNN Confusion')