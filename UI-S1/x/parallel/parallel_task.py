from functools import partial
from squirrel.iterstream import IterableSource
try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm
'''
IterableSource 只支持单参数输入 因此对于用户的多参数组合输入 我们先把它们pack到一个tuple中 同时hack用户传入的function_ 先进行参数拆解
'''
def gen_with_indices(data_):
    for index, line in enumerate(zip(*data_)):
        yield line+(index,)

def gen(data_):
    for index, line in enumerate(zip(*data_)):
        yield line

def exec(args, function_):
    return function_(*args)

class ParallelTask():
    def __init__(self, data_, function_, total=None, num_process=81, passing_indices=False, return_list=True):
        function_ = partial(exec, function_=function_)
        if passing_indices:
            data_ = gen_with_indices(data_)
        else:
            data_ = gen(data_)
        self.t = IterableSource(data_)
        self.total = total
        self.num_process = num_process
        self.function_ = function_
        self.return_list = return_list

    def _collect_list(self, handler, tqdm_args):
        collections_ = []
        for flag, line in tqdm(handler, **tqdm_args):
            if flag:
                collections_.append(line)
        return collections_
    
    def _collect_genetrator(self, handler, tqdm_args):
        for flag, line in tqdm(handler, **tqdm_args):
            if flag:
                yield line

    def run_and_collect(self, tqdm_args=None, buffer=None):
        if tqdm_args is None:
            tqdm_args = {}
        if buffer is None:
            buffer = max(100, self.num_process*3)
        handler = self.t.async_map(self.function_, max_workers=self.num_process, buffer=buffer)
        
        if self.total is not None and 'total' not in tqdm_args:
            tqdm_args['total'] = self.total
        if self.return_list:
            return self._collect_list(handler, tqdm_args)
        else:
            return self._collect_genetrator(handler, tqdm_args)