"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
import random
from util import *
from collections import Counter
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n

        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        self.probabilities_ = dict()
        n,d = X.shape
        self.probabilities_["0"] = 0
        self.probabilities_["1"] = 0
        for item in y:
            if(item == 0):
                self.probabilities_["0"]+=1
            else:
                self.probabilities_["1"]+=1
        self.probabilities_["0"] = self.probabilities_["0"]/n
        self.probabilities_["1"] = self.probabilities_["1"]/n
        ### ========== TODO : END ========== ###

        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        n,d = X.shape
        y = np.random.choice(2, n, p=[self.probabilities_["0"], self.probabilities_["1"]])
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0
    test_error = 0
    for i in range(ntrials):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(x_train, y_train)
        y_pred_train = clf.predict(x_train)
        y_pred_test = clf.predict(x_test)
        train_err = 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        train_error += train_err
        test_err = 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)
        test_error += test_err
    ### ========== TODO : END ========== ###

    return train_error/ntrials, test_error/ntrials


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    #for i in range(d) :
    #    plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    majority_clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    majority_clf.fit(X, y)                  # fit training data using the classifier
    y_pred = majority_clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    random_clf = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    random_clf.fit(X, y)
    y_pred = random_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    tree_clf = DecisionTreeClassifier("entropy")
    tree_clf.fit(X,y)
    y_pred = tree_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    # save the classifier -- requires GraphViz and pydot
    from io import StringIO
    import pydotplus
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(tree_clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    clf_3 = KNeighborsClassifier(n_neighbors=3)
    clf_5 = KNeighborsClassifier(n_neighbors=5)
    clf_7 = KNeighborsClassifier(n_neighbors=7)
    clf_3.fit(X, y)
    clf_5.fit(X, y)
    clf_7.fit(X, y)
    y_pred_3 = clf_3.predict(X)
    y_pred_5 = clf_5.predict(X)
    y_pred_7 = clf_7.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred_3, normalize=True)
    print('\t-- training error for 3-neighbors: %.3f' % train_error)
    train_error = 1 - metrics.accuracy_score(y, y_pred_5, normalize=True)
    print('\t-- training error for 5-neighbors: %.3f' % train_error)
    train_error = 1 - metrics.accuracy_score(y, y_pred_7, normalize=True)
    print('\t-- training error for 7-neighbors: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    #MajorityVoteClassifier
    train_error, test_error = error(majority_clf, X, y)
    print("MajorityVoteClassifier has training error: ", train_error, "and test error: ", test_error)
    #Random Classifier
    train_error, test_error = error(random_clf, X, y)
    print("RandomClassifer has training error: ", train_error, "and test error: ", test_error)
    #Decision Tree
    train_error, test_error = error(tree_clf, X, y)
    print("Decision Tree has training error: ", train_error, "and test error: ", test_error)
    #KNN-5
    train_error, test_error = error(clf_5, X, y)
    print("KNN-5 has training error: ", train_error, "and test error: ", test_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    scores = list()
    k_values = list()
    for k in range(1,50,2):
        k_values.append(k)
        clf = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(clf, X, y, cv=10)
        scores.append(np.mean(score))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("k, number of neighbors")
    plt.ylabel("score")
    plt.savefig("4fgraph.pdf")
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    max_depths = list()
    train_errs = list()
    test_errs = list()
    for d in range(1, 21):
        tree_clf = DecisionTreeClassifier("entropy", max_depth=d)
        train_err, test_err = error(tree_clf, X, y)
        max_depths.append(d)
        train_errs.append(train_err)
        test_errs.append(test_err)
    plt.clf()

    plt.plot(max_depths, train_errs, '-b', label='Train Error', marker="o")
    plt.plot(max_depths, test_errs, '-r', label='Test Error', marker="o")
    plt.legend(loc='lower left')
    plt.xlabel("Max depth for Tree classifier")
    plt.ylabel("Error rates")
    plt.savefig("4ggraph.pdf")
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
    index = []
    dtTrainErrs = []
    dtTestErrs = []
    knnTrainErrs = []
    knnTestErrs = []
    plt.clf()
    tree_clf = DecisionTreeClassifier("entropy", max_depth=3)
    for i in range(1, 11):
        index.append(i/10)
        tree_train_err = 0
        tree_test_err = 0
        knn_train_err = 0
        knn_test_err = 0

        for j in range(100):
            X_train_inner, _, y_train_inner, _= train_test_split(X_train, y_train, test_size=(1-i*0.1), random_state=i)

            tree_clf.fit(X_train_inner, y_train_inner)
            clf_7.fit(X_train_inner, y_train_inner)
            tree_pred = tree_clf.predict(X_train_inner)
            tree_train_err += 1 - metrics.accuracy_score(y_train_inner, tree_pred, normalize=True)
            tree_pred = tree_clf.predict(X_test)
            tree_test_err += 1 - metrics.accuracy_score(y_test, tree_pred, normalize=True)

            knn_pred =clf_7.predict(X_train_inner)
            knn_train_err += 1 - metrics.accuracy_score(y_train_inner, knn_pred, normalize=True)
            knn_pred = clf_7.predict(X_test)
            knn_test_err += 1 - metrics.accuracy_score(y_test, knn_pred, normalize=True)

        dtTrainErrs.append(tree_train_err/100)
        dtTestErrs.append(tree_test_err/100)

        knnTrainErrs.append(knn_train_err/100)
        knnTestErrs.append(knn_test_err/100)

    plt.plot(index, dtTrainErrs, label='DT Train Error', marker='o')
    plt.plot(index, dtTestErrs, label='DT Test Error', marker='o')
    plt.plot(index, knnTrainErrs, label='KNN Train Error', marker='o')
    plt.plot(index, knnTestErrs, label='KNN Test Error', marker='o')
    plt.legend(loc='upper right')
    plt.xlabel('% of data used for training')
    plt.ylabel("Error rate")
    plt.savefig("4hgraph.pdf")
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
