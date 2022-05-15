import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from IPython.display import display
import time

# download dataset
nltk.download('treebank')
nltk.download('universal_tagset')
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))

# split data
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
RANDOM_STATE = 101
train_set, test_set = train_test_split(nltk_data, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# create list of tagged words
train_tagged_words, test_tagged_words = [], []
for sentence in train_set:
    for word in sentence:
        train_tagged_words.append(word)
for sentence in test_set:
    for word in sentence:
        test_tagged_words.append(word)
tags = set()
for word, tag in train_tagged_words:
    tags.add(tag)

# compute emission probability
def word_given_tag(word, tag, data=train_tagged_words):
    number_of_tag, number_of_word_given_tag = 0, 0
    for pair in data:
        if pair[1] == tag:
            number_of_tag += 1
            if pair[0] == word:
                number_of_word_given_tag += 1
    return (number_of_word_given_tag, number_of_tag)

# compute Transition Probability
def tag2_before_tag1(tag2, tag1, data=train_tagged_words):
    tags = []
    number_of_tag1, number_of_tag2_tag1 = 0, 0
    for word, tag in data:
        tags.append(tag)
        if tag == tag1:
            number_of_tag1 += 1
    for i in range(len(tags)-1):
        if tags[i] == tag1 and tags[i+1] == tag2:
            number_of_tag2_tag1 += 1
    return (number_of_tag2_tag1, number_of_tag1)

# create transition probability matrix
tags_transition_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, tag1 in enumerate(list(tags)):
    for j, tag2 in enumerate(list(tags)):
        number_of_tag2_tag1, number_of_tag1 = tag2_before_tag1(tag2, tag1)
        tags_transition_matrix[i, j] = number_of_tag2_tag1 / number_of_tag1

# put in pandas
tags_df = pd.DataFrame(tags_transition_matrix, columns=list(tags), index=list(tags))
print("The Transition Probability Matrix")
display(tags_df)
print()

# Viterbi algorithm
def Viterbi(sentence, data=train_tagged_words):
    state = []
    global tags
    T = list(tags)
    for key, word in enumerate(sentence):
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
            number_of_word_given_tag, number_of_tag = word_given_tag(word, tag)
            emission_p = number_of_word_given_tag / number_of_tag
            state_probability = emission_p * transition_p
            p.append(state_probability)
        pmax = max(p)
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(sentence, state))

# testing
def run_part_of_test(number_of_test=100, seed=101):
    random.seed(seed)
    random_state = [random.randint(0, len(test_set) - 1) for x in range(number_of_test)]
    test_run = [test_set[i] for i in random_state]
    return test_run
    
test_run = run_part_of_test()
# test_run = test_set  # test all the test set
test_tagged_words, test_untagged_words = [], []
for sentence in test_run:
    for word, tag in sentence:
        test_tagged_words.append((word, tag))
        test_untagged_words.append(word)
# run Viterbi on test set
print("The Viterbi Algorithm is running...")
start = time.time()
prediction_tagged_words = Viterbi(test_untagged_words)
end = time.time()
print("The Viterbi Algorithm takes {:.2f} seconds to run.".format(end - start))

# check accuracy
correct = 0
for i, j in zip(prediction_tagged_words, test_tagged_words):
    if i == j:
        correct += 1
accuracy = correct/len(prediction_tagged_words)
print('The accuracy of Viterbi Algorithm is {:.2f}%'.format(accuracy*100))
