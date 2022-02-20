# Wordle Bot
Wordle bot inspired by 3B1B video. Uses multiprocessing to accelerate calculations.

## Running
Choose some constants for simulation parameters
```
MAX_WORD_LIST_LENGTH = 200
N_MAX_TRIES = 5
INITIAL_GUESS = "slate" # Word to initially guess. Keep blank for initial entropy calculation. '?' for random 
```
Run simulation
```
python wordle_bot.py
```

![Example simulation](https://raw.githubusercontent.com/caelan-a/wordle_bot/main/example.PNG)

## Multiprocessing
This project uses the `multiprocessing` module shipped with the python interpreter for distributed calculations.
Can see the effect of multiprocessing on a test function in `multiprocessing_test.py`
Precalculated results in `multiprocessing_test_results.txt`