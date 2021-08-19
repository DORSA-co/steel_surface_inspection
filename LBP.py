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
    h, w = image.shape
    # divide image into n*n subimage
    n = 8
    dimx = int(h / n)
    dimy = int(w / n)
    # calculate weight for each subimage: w_i = sum(g(x,y)^2)/sg - (sum(g(x, y))/sg)^2 for each x, y, where g is subimage and sg is dimension of subimage
    weight = np.zeros(n ** 2)
    c = 0
    for i in range(dimx, h + 1, dimx):
        for j in range(dimy, w + 1, dimy):
            Re = image[i - dimx:i, j - dimy:j]
            Re2 = Re ** 2
            weight[c] = (Re2.sum() / (dimx * dimy)) - (Re.sum() / (dimx * dimy)) ** 2
            c += 1
    # normalize weight between [0, 1]
    weight = (weight - np.min(weight)) / np.ptp(weight)
    c = 0
    if method == 'uniform':
        bins = P + 2
    else:
        bins = int(math.pow(2, P))
    hist = np.zeros(bins, dtype='int64')
    # calculate lbp histogram for each subimage if it's weight is more than average of weights
    for i in range(dimx, h + 1, dimx):
        for j in range(dimy, w + 1, dimy):
            if weight[c] >= 0.85:
                lbp = feature.local_binary_pattern(image[i - dimx:i, j - dimy:j], P, R, method)
                h, _ = np.histogram(lbp.ravel(), density=False, bins=np.arange(0, bins + 1), range=(0, bins))
                hist = np.add(hist, h)
            c += 1
    queue.put({
        'patterns': hist
    })


def multiPreocessing(image, P, R, method, num_processes):
    # Spawn the processes
    height, width = image.shape
    processes = []
    queue = Queue()
    n = math.floor(math.sqrt(num_processes))
    segment_height = int(np.floor(height / n))
    segment_width = int(np.floor(width / n))
    for i in range(n):
        left_bound = i * segment_height
        right_bound = (i * segment_height) + segment_height
        if i == (num_processes - 1):
            # The last process should also process any remaining rows
            right_bound = height
        for j in range(n):
            up_bound = j * segment_width
            down_bound = (j * segment_width) + segment_width
            if j == (num_processes - 1):
                # The last process should also process any remaining rows
                down_bound = width
            I = image[left_bound:right_bound, up_bound:down_bound]
            process = Process(target=get_lbp, args=(I, P, R, method, queue))
            process.start()
            processes.append(process)

    # Wait for all processes to finish
    results = [queue.get() for process in processes]
    [process.join() for process in processes]

    # Format the pixels correctly for the output function,
    # which expects a linear list of pixel values.
    if method == 'uniform':
        s = P + 2
    else:
        s = int(math.pow(2, P))
    hist = np.zeros(s).astype('int64')
    for result in results:
        hist = np.add(hist, result['patterns'])
        # hist = hist + result['patterns']
    return hist


image = imread('/home/reyhane/Desktop/b.jpg', 0)
start_time = timeit.default_timer()
hist = multiPreocessing(image, 4, 1, 'uniform', 9)
end_time = timeit.default_timer()
print(end_time - start_time)
