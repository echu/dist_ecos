import multiprocessing


class PersistProx(multiprocessing.Process):
    def __init__(self, local_problem, inbox, outbox):
        '''sets up prox function. we assume the cones are linear.
           we expect numpy/scipy data '''
        super(PersistProx, self).__init__()

        self.local_problem = local_problem

        self.inbox = inbox
        self.outbox = outbox

    def run(self):
        while True:
            next_arg = self.inbox.get()
            if next_arg is None:
                # Poison pill means we should exit
                print '%s: Exiting' % self.name
                break
            x, info = self.local_problem.xupdate(next_arg)
            self.outbox.put((x, info))
            self.inbox.task_done()
        return
