#!/usr/bin/python

import multiprocessing as mp
from multiprocessing import Array, Pool, Process, Value
from queue import Queue
from time import time

import numpy


def operationForPool(elements, min=2, max=6):
    sum = 0
    for e in elements:
        if min <= e <= max:
            sum += 1 
    return sum

def operation(elements, min=2, max=6):
    sum = 0
    for i in elements:
        for e in i:
            if min <= e <= max:
                sum += 1 
    return sum

def parallelisableOperation(elements, ret_values, index, min=2, max=6):
    ret_values[index] = operation(elements, min=min, max=max)

if __name__ == '__main__':
    with open("results.txt", "a") as file_object:
        file_object.write(f"n_elements, time_taken_raw_mp (s), time_taken_raw_mp_pool (s), time_taken_single (s)\n")

    for j in range(2,9):
        elements = numpy.random.randint(0, 10, (10**j, 10))

        n_processes = mp.cpu_count()

        if n_processes > len(elements): raise Exception("n_processes must be greater than number of elements")
        batch_size = int(len(elements)/n_processes)
    
        print(f"n processes: {n_processes}")
        print(f"batch size: {batch_size}")


        print("Using raw multiprocessing:")
        start = time()

        pqueue = Queue()
        ret_values = Array("i", [0]*n_processes)

        for i in range(0,n_processes):
            if(i == n_processes-1):
                pqueue.put(elements[i*batch_size:])
            else:
                pqueue.put(elements[i*batch_size:(i+1)*batch_size])

        processes = [Process(target=parallelisableOperation, args=(pqueue.get(), ret_values, i)) for i in range(0,n_processes)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        end = time()

        total_sum = sum(ret_values)

        print(f"Number of elements between 2 and 6: {total_sum}")
        time_taken_raw_mp = round(end-start,2)
        print(f"Time taken: {time_taken_raw_mp}s")

        print("Using multiprocessing pool:")
        start = time()
        with Pool(n_processes) as p:
            total_sum = sum(p.map(operationForPool, elements))
        end = time()

        print(f"Number of elements between 2 and 6: {total_sum}")
        time_taken_raw_mp_pool = round(end-start,2)
        print(f"Time taken: {time_taken_raw_mp_pool}s")
        
        print("Using single process:")
        start = time()
        total_sum = operation(elements)
        end = time()
        print(f"Number of elements between 2 and 6: {total_sum}")
        time_taken_single = round(end-start,2)
        print(f"Time taken: {time_taken_single}s")

        with open("results.txt", "a") as file_object:
            file_object.write(f"{10**j}, {time_taken_raw_mp}, {time_taken_raw_mp_pool}, {time_taken_single}\n")

