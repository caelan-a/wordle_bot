from concurrent.futures import process
from ctypes import c_char, c_char_p
import ctypes
from multiprocessing.dummy import Array, Process
from queue import Queue
from random import choice
import sys
import time
from unittest import result
import numpy as np
from termcolor import colored, cprint
import itertools
import multiprocessing as mp

MAX_WORD_LIST_LENGTH = 2500
N_MAX_TRIES = 5
INITIAL_GUESS = "adieu" # Word to initially guess. Keep blank for initial entropy calculation. '?' for random 

DOESNT_CONTAIN_LETTER = "0"
CONTAINS_LETTER_AT_POSITION = "G"
CONTAINS_LETTER = "Y"

def getMatchPermutations():
    match_permutations = list(itertools.product([DOESNT_CONTAIN_LETTER, CONTAINS_LETTER, CONTAINS_LETTER_AT_POSITION], repeat=5))
    # Remove impossible matches
    match_permutations = [m for m in match_permutations if not (m.count(CONTAINS_LETTER) == 1 and m.count(CONTAINS_LETTER_AT_POSITION) == 4) ] 
    return match_permutations

MATCH_PERMUTATIONS = getMatchPermutations()
    
def isGuessCorrect(guess_result):
    return guess_result.count(CONTAINS_LETTER_AT_POSITION) == len(guess_result)

def getWordsMatchingGuessResult(words, guess, guess_result):
    matching_words = []
    for word in words:
        w = list(word)
        skip = False
        for i in range(0, len(guess_result)):
            if guess_result[i] == CONTAINS_LETTER_AT_POSITION:
                if guess[i] != w[i]:
                    skip = True
            elif guess_result[i] == CONTAINS_LETTER:
                if guess[i] in w and guess[i] != w[i]:
                    w[w.index(guess[i])] = 0
                else:
                    skip = True
            elif guess_result[i] == DOESNT_CONTAIN_LETTER:
                if guess[i] in w:
                    skip = True
        
        if not skip:
            matching_words.append(word)
    return [guess_result, matching_words]

def printGuess(guess, guess_result):
    colored_letters = []

    for i in range(0, len(guess)):
        if guess_result[i] == CONTAINS_LETTER_AT_POSITION:
            color = "on_green"
        elif guess_result[i] == CONTAINS_LETTER:
            color = "on_yellow"
        else:
            color = "on_red"

        colored_letters.append(colored("-", "white", color))
    print(*colored_letters)

def compareWords(guess,target):
    if(len(guess) != len(target)):
        raise Exception("Words must be of same length")

    guess = list(guess)
    target = list(target)

    result = [0 for i in range(0, len(target))]

    for i in range(0, len(target)):
        if guess[i] == target[i]:
            result[i] = CONTAINS_LETTER_AT_POSITION
        elif guess[i] in target:
            result[i] = CONTAINS_LETTER

            target[target.index(guess[i])] = 0
        else:
            result[i] = DOESNT_CONTAIN_LETTER

    return result

def getWords():
    f = open("wordle_data/possible_words.txt", "r")
    words = f.read().splitlines()

    if(len(words) > MAX_WORD_LIST_LENGTH):
        return words[:MAX_WORD_LIST_LENGTH]
    else:
        return words

def getProbability(remaining_words, all_words):
    return len(remaining_words)/len(all_words)

def getInformationEntropy(probability):
    if(probability == 0):
        raise Exception("Probability can't be 0")
    return -np.math.log2(probability)
     
def getExpectedInformation(word, words_all):
    entropies = []

    words_matching_guess_results = [getWordsMatchingGuessResult(words_all, word, guess_result) for guess_result in MATCH_PERMUTATIONS]
    probabilities = [getProbability(words[1], words_all) for words in words_matching_guess_results]

    # Ignore null probabilities corresponding to impossible guess_results 
    entropies = [[p, getInformationEntropy(p)] for p in probabilities if p != 0]
    avg_information_entropy = sum([e[0] * e[1] for e in entropies])

    return avg_information_entropy

def getBestGuess(words, words_all):
    
    res = [getExpectedInformation(word, words_all) for word in words]

    max_expected_info = max(res)
    best_guess = words[res.index(max_expected_info)]

    # print(f"best_guess: {best_guess}, max_expected_info: {max_expected_info}")
    return (best_guess, max_expected_info)

def parallelisableOperation(func, elements, result_dict, index):
    if(len(elements[0]) == 0): return 
    
    try:
        res = func(*elements)
        result_dict[res[0]] = res[1]
    except:
        print(f"Failed to run function on elements: {elements}")

def getBestGuessInParallel(possible_words):
    if len(possible_words) == 0:
        raise Exception("Possible words must not be empty")

    start_time = time.time()

    n_processes = mp.cpu_count()

    batch_size = int(len(possible_words)/n_processes)
    pqueue = Queue()

    manager = mp.Manager()
    result_dict = manager.dict()


    for i in range(0,n_processes):
        if(i == n_processes-1):
            pqueue.put(possible_words[i*batch_size:])
        else:
            pqueue.put(possible_words[i*batch_size:(i+1)*batch_size])

    processes = [Process(target=parallelisableOperation, args=(getBestGuess, (pqueue.get(),possible_words), result_dict, i)) for i in range(0,n_processes)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    max_entropy = max(result_dict.values())
    best_guess = result_dict.keys()[result_dict.values().index(max_entropy)]
    end_time = time.time()

    print(f"Determined best guess: {best_guess} (I_avg = {round(max_entropy,2)} bits) in {round(end_time - start_time,2)}s")
    return best_guess, max_entropy

if __name__ == '__main__':
    remaining_possible_words = getWords()
    words_tried = []
    print(f"\nWordle Bot v1.0")
    print(f"Number of words: {len(remaining_possible_words)}")
    target = choice(remaining_possible_words)
    print(f"Target word: {target}")

    for i in range(1,N_MAX_TRIES+1):
        print(f"\nGuess {i}")
        print(f"Remaining possibilities: {len(remaining_possible_words)}")
        print(f"Remaining information: {round(np.math.log2(len(remaining_possible_words)),2)} bits")

        print(f"Words tried: {words_tried}")
       
        if(INITIAL_GUESS != "" and  i == 1):
            if INITIAL_GUESS == "?":
                guess = choice(remaining_possible_words)
                print(f"Random initial guess: {guess}")
            else:
                guess = INITIAL_GUESS
                print(f"Chosen initial guess: {guess}")
        else:
            guess, entropy = getBestGuessInParallel(remaining_possible_words)

        guess_result = compareWords(guess, target)

        printGuess(guess, guess_result)
        if(isGuessCorrect(guess_result)):
            print("Guessed word correctly!")
            exit()
        else:
            print("Incorrect guess!")
            words_tried.append(guess)
            remaining_possible_words = getWordsMatchingGuessResult(remaining_possible_words, guess, guess_result)[1]
    
    print(f"Failed to guess word after {N_MAX_TRIES}")