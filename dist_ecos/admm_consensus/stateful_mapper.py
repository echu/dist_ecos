from multiprocessing import Process, JoinableQueue, Queue
#TODO: I think there's a race condition concerning the queues.
#programs with fast proxes and many iterations have a chance of stalling...


class StatefulMapper(object):
    """ Maintain a list of python objects whose state should persist after calling
        the map function. We abuse functional programming constructs by allowing
        the mapping function to modify each object's state. It is also
        possible to pass in additional data with each map call.

        The module is intended to be used for parallel computations. Each object
        from the initializing list is given its own process, which persists
        until close() is called. The function
        evaluations in a map() call are done on separate processes. Computations
        can also be done serially, for comparison.

        Note that all data and functions involved in a map() call must be
        pickleable. The results will be unpickled objects sent
        from each process.

    """
    def __init__(self, obj_list, parallel=True):
        if parallel:
            self.list = ParallelList(obj_list)
        else:
            self.list = SerialList(obj_list)

    def map(self, func, *args, **kwargs):
        """
            Should be equivalent to the following python list comprehension:
            [func(x, *args, **kwargs) for x in obj_list]
        """
        return self.list.map(func, *args, **kwargs)

    def close(self):
        """
            Close the processes wich are holding the objects.
        """
        self.list.close()


class ProcWrapper(Process):
    def __init__(self, obj, inbox, outbox):
        super(ProcWrapper, self).__init__()
        self.inbox = inbox
        self.outbox = outbox
        self.obj = obj

    def run(self):
        while True:
            msg = self.inbox.get()

            if msg is None:
                #end the process
                break

            func, args, kwargs = msg
            result = func(self.obj, *args, **kwargs)
            self.outbox.put(result)
            self.inbox.task_done()
            #wait for all others to complete before you try to take again
            self.inbox.join()


class ParallelList(object):
    def __init__(self, obj_list):
        self.inbox = JoinableQueue()
        self.outbox = Queue()
        self.wrappers = []
        for obj in obj_list:
            wrapper = ProcWrapper(obj, self.inbox, self.outbox)
            self.wrappers.append(wrapper)
            wrapper.start()

    def map(self, func, *args, **kwargs):
        result = []
        for x in self.wrappers:
            self.inbox.put((func, args, kwargs))
        self.inbox.join()  # wait for each process to finish computation

        for x in self.wrappers:
            result.append(self.outbox.get())

        return result

    def close(self):
        #send poison pill
        for x in self.wrappers:
            self.inbox.put(None)

        for x in self.wrappers:
            x.join()


class SerialList(object):
    def __init__(self, obj_list):
        self.obj_list = obj_list

    def map(self, func, *args, **kwargs):
        return [func(obj, *args, **kwargs) for obj in self.obj_list]

    def close(self):
        pass


if __name__ == "__main__":
    import numpy as np

    def add_normalize(cur, toadd):
        cur += toadd
        cur /= np.linalg.norm(cur)
        return cur

    N = 100  # subsystems
    n = 100  # vector length
    iterations = 100

    #array_list = [np.random.randn(n) for i in xrange(N)]

    # serial code
    #print "serial code"
    #for i in xrange(iterations):
    #    print '>> iteration: ', i
    #    toadd = np.random.randn(n)
    #    [add_normalize(x, toadd) for x in array_list]
    #
    #    for vec in array_list:
    #        print vec


    # parallel code
    print "parallel code"
    array_list = [np.random.randn(n) for i in xrange(N)]
    a = StatefulMapper(array_list, kind='parallel')

    for i in xrange(iterations):
        print '>> iteration: ', i
        toadd = np.random.randn(n)
        result = a.map(add_normalize, toadd)

    a.close()
