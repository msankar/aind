import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = [] #list of dicts, key word = value logL
    guesses = []

    for sequence, length in test_set.get_all_Xlengths().values():
        probabilityDict = {}
        maxWord = None
        maxScore = float("-inf")
        for word, model in models.items():
            #print(word, model)
            try:
                score = model.score(sequence, length)
                probabilityDict[word] = score
                if score > maxScore:
                    maxWord = word
                    maxScore = score
            except:
                probabilityDict[word] = float("-inf")
                pass
        guesses.append(maxWord)
        probabilities.append(probabilityDict)
    return probabilities, guesses



