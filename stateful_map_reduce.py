#stateful map reduce

from multiprocessing import Process, JoinableQueue, Queue


class ProcWrapper(Process):
    def __init__(self, obj, inbox, outbox):
        super(ProcWrapper, self).__init__()
        self.inbox = inbox
        self.outbox = outbox
        self.obj = obj

    def run(self):
        while True:
            func = self.inbox.get()
            if func is None:
                break
            result = func(self.obj)
            self.outbox.put(result)
            self.inbox.task_done()
            #wait for all others to complete before you try to take again
            self.inbox.join()


class ProcList(object):
    def __init__(self, obj_list):
        self.inbox = JoinableQueue()
        self.outbox = Queue()
        self.wrappers = []
        for obj in obj_list:
            wrapper = ProcWrapper(obj, self.inbox, self.outbox)
            self.wrappers.append(wrapper)
            wrapper.start()

    def map(self, func):
        result = []
        for x in self.wrappers:
            self.inbox.put(func)
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

if __name__ == "__main__":

    def add2(x):
        val = x[-1]+2
        x.append(val)
        return val

    def ident(x):
        return x

    def last(x):
        return x[-1]

    def double(x):
        x += x
        #don't return anything. just change state

    a = ProcList([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    print a.map(ident)
    print a.map(add2)
    print a.map(ident)
    print a.map(last)
    print a.map(double)
    print a.map(ident)
    a.close()
