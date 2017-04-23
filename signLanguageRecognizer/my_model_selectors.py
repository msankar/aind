import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    Reference: ftp://metron.sta.uniroma1.it/RePEc/articoli/2002-LX-3_4-11.pdf
        N: number of data points
        p: number of parameters
        d: number of features
        n: number of HMM states
        p = n*(n-1) + (n-1) + 2*d*n = n^2 + (2nd) - 1
            No. of probabilities in transition matrix +
            No. of probabilities in initial distribution +
            No. of Gaussian mean +
            No. of Gaussian variance
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        lowestBICScore = None
        lowestBICScoreModel = None

        for n_component in range(self.min_n_components, self.max_n_components+1):
            try:
                # Fit model using n compoments
                model = self.base_model(n_component)
                logL = model.score(self.X, self.lengths)

                d = model.n_features
                p = n_component ** 2 + (2 * n_component * d) - 1
                bic_score = (-2 * logL + p * math.log(n_component))
                if lowestBICScore is None or lowestBICScore > bic_score:
                    lowestBICScore = bic_score
                    lowestBICScoreModel = model
            except:
                pass
        return lowestBICScoreModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        highestDICScore = None
        bestModel = None

        otherWords = list(self.words)
        otherWords.remove(self.this_word)

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n) # Fits HMM model
                score = model.score(self.X, self.lengths) #Get the score

                sumOtherWordsScore = 0.0
                # For each word that is not this word, fit model and score.
                for otherWord in otherWords:
                    X, lengths = self.hwords[otherWord]
                    sumOtherWordsScore += model.score(X, lengths)

                # Calculate DIC Score
                dic =  score - (sumOtherWordsScore / (len(self.words) - 1))

                # Keep model with higher DIC score
                if highestDICScore is None or highestDICScore < dic:
                    highestDICScore = dic
                    bestModel = model
            except:
                    pass

        return bestModel

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        Ref: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        splitMethod = KFold()
        allNComponents = []
        allScores = []
        for nComponents in range(self.min_n_components, self.max_n_components+1):
            try:
                if len(self.sequences) > 2: # Make sure we can perform a split.
                    scores = []
                    for trainIndexList, testIndexList in splitMethod.split(self.sequences):
                        # Prepare training sequences
                        self.X, self.lengths = combine_sequences(trainIndexList, self.sequences)
                        # Prepare testing sequences
                        X_test, lengths_test = combine_sequences(testIndexList, self.sequences)
                        model = self.base_model(nComponents)
                        score = model.score(X_test, lengths_test)
                        scores.append(score)
                    allScores.append(np.mean(scores))
                else:
                    model = self.base_model(nComponents)
                    score = model.score(self.X, self.lengths)
                    allScores.append(score)
                allNComponents.append(nComponents)
            except:
                pass

        bestComponents = allNComponents[np.argmax(allScores)] if allScores else self.n_constant
        return self.base_model(bestComponents)
