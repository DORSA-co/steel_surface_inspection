import math

from cv2 import *
import numpy as np
import timeit
from skimage import feature
from multiprocessing import Process, Queue

def get_lbp(image, P, R, method, queue):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, P, R, method)
    bin_max = lbp.max() + 2
    range_max = lbp.max() + 1
    hist, _ = np.histogram(lbp.ravel(), density=False, bins=np.arange(0, bin_max), range=(0, range_max))
    queue.put({
        'patterns': hist
    })


def multiPreocessing(image, P, R, method, num_processes):
    # Spawn the processes
    height, width = image.shape
    processes = []
    queue = Queue()
    segment_height = int(np.floor(height / num_processes))
    for process_id in range(num_processes):
        left_bound = process_id * segment_height
        right_bound = (process_id * segment_height) + segment_height
        if process_id == (num_processes - 1):
            # The last process should also process any remaining rows
            right_bound = height
        I = image[left_bound:right_bound, ]
        process = Process(target=get_lbp, args=(I, P, R, method, queue))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    results = [queue.get() for process in processes]
    [process.join() for process in processes]

    # Format the pixels correctly for the output function,
    # which expects a linear list of pixel values.
    if method=='uniform':
        s = P + 2
    else:
        s = int(math.pow(2, P))
    hist = np.zeros(s).astype('int64')
    for result in results:
        hist = np.add(hist, result['patterns'])
        #hist = hist + result['patterns']
    return hist

image = imread('/home/reyhane/Desktop/a.jpg', 0)
start_time = timeit.default_timer()
hist = multiPreocessing(image, 8, 1, 'uniform', 8)
print(hist)
end_time = timeit.default_timer()
print(end_time - start_time)
