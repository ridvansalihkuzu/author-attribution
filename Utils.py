from collections import defaultdict
import os
import codecs
import numpy as np
import math

class Utils:
    @staticmethod
    def count_ngrams(document, n, normalise=False):
        counts = defaultdict(float)  # Default to 0.0 if the key not found
        # Iterate through all n-grams in the document
        for i in range(len(document) - n + 1):
            ngram = document[i:i + n]
            # Update the count of this n-gram.
            counts[ngram] = counts[ngram] + 1
        if normalise:
            # Normalise so that sums equal 1
            normalise_factor = float(len(document) - n + 1)
            for ngram in counts:
                counts[ngram] /= normalise_factor
        return counts

    @staticmethod
    def remove_linebreaks(document):
        doc=document.replace("\\s+", " ")

        return doc

    @staticmethod
    def extract_ngrams(vocabulary,document, n):

        grams = list()  # Default to 0.0 if the key not found
        # Iterate through all n-grams in the document
        for i in range(len(document) - n + 1):
            ngram = document[i:i + n]
            try:
                ind=vocabulary.index(ngram)
            except:
                pass
            else:
                grams.append(ngram)
        return grams

    @staticmethod
    def get_corpus(folder, concat=1):
        documents = []
        authors = []
        training_mask = []
        authornum = 0
        i = 0
        subfolders = [name for name in os.listdir(folder)
                      if os.path.isdir(os.path.join(folder, name))]
        for subfolder in subfolders:
            sf = os.path.join(folder, subfolder)
            print("Author %d is %s" % (authornum, subfolder))
            listdir=os.listdir(sf)
            sample_n=math.floor(len(listdir)/concat)
            for i in range(0,sample_n,1):
                concat_text=""
                for j in range(0,concat,1):
                    ind=i*concat+j
                    if(len(listdir)>ind):
                        with codecs.open(os.path.join(sf, listdir[ind]), encoding='latin-1') as input_f:
                            concat_text=concat_text+" "+Utils.cleanFile(input_f.read())

                documents.append(concat_text)
                authors.append(authornum)
                training_mask.append(True)

            authornum += 1

        min_docs = 1
        c = np.bincount(authors)
        validauthors = [authornum for authornum, count in enumerate(c) if count >= min_docs]
        documents = [d for d, c in zip(documents, authors) if c in validauthors]
        authors = [c for c in authors if c in validauthors]
        assert len(documents) == len(authors)
        return documents, np.array(authors, dtype='int')

    @staticmethod
    def cleanFile(document):
        lines = document.split("\n")
        start = 0
        end = len(lines)
        for i in range(len(lines)):
            line = lines[i]
        return "\n".join(lines[start:end])

    @staticmethod
    def vector_space_model(author_profiles, dictionary=None):
        if dictionary is not None:
            dictionary = Utils.top_L(dictionary, 20000)
        else:
            dictionary = set()
            for i in range(len(author_profiles)):
                dictionary.update(author_profiles[i].keys())

        vector_space = np.zeros((len(author_profiles), len(dictionary)))
        for i in range(len(author_profiles)):
            author = author_profiles[i]
            author_vect = np.array([author.get(dic, 0.) for dic in dictionary])
            for j in range(len(dictionary)):
                vector_space.itemset((i, j), author_vect[j])

        return vector_space, dictionary



    @staticmethod
    def vector_space_represent(author_profile, dictionary):
        author_vect = np.array([author_profile.get(dic, 0.) for dic in dictionary])
        return author_vect

    @staticmethod
    def create_dictionary(documents, n):
        # Creates a profile of a document or list of documents.
        if isinstance(documents, str):
            # documents can be either a list of documents, or a single document.
            # if it's a single document, convert to a list
            documents = [documents, ]
        # profile each document independently
        profiles = (Utils.count_ngrams(document, n, normalise=False)
                    for document in documents)
        # Merge the profiles
        main_profile = defaultdict(float)
        for profile in profiles:
            for ngram in profile:
                main_profile[ngram] += profile[ngram]
        # Normalise the profile
        num_ngrams = sum(main_profile.values())
        for ngram in main_profile:
            main_profile[ngram] /= num_ngrams
        # Return the profile with only the top L n-grams
        return Utils.top_L(main_profile,10000) # Most frequent 10000 terms in dictionary

    @staticmethod
    def top_L(profile, L):
        # Returns the profile with only the top L most frequent n-grams
        if L >= len(profile):
            return profile
        threshold = sorted(map(abs, profile.values()))[-L]

        copy = defaultdict(float)
        for key in profile:
            if abs(profile[key]) >= threshold:
                copy[key] = profile[key]
        return copy

    @staticmethod
    def read_double_array(filename):
        myArray = []
        textFile = open(filename)
        lines = textFile.readlines()
        for line in lines:
            myArray.append(float(line.replace("\n","")))

        return myArray

