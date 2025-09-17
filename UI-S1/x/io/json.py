from functools import partial
import inspect
import mmap
from pathlib import Path
import json
import pickle
import warnings
import traceback
import json5

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def smart_json_loads(line):
    try:
        try:
            tmp = json.loads(line.strip())
            return tmp
        except KeyboardInterrupt:
            raise
        except:
            tmp = json5.loads(line.strip())
            return tmp
    except:
        print(f'Invalid Json Line [Json Start]{line}[Json End]')
        traceback.print_exc()
    return None

def _read_json_generator(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = smart_json_loads(line)
            if tmp:
                yield tmp


def _read_json_list_generator(paths):
    for path in paths:
        yield from read_json(path, generator_mode=True)
        
def read_json_list(paths, generator_mode=False):
    if generator_mode:
        return _read_json_list_generator(paths)
    else:
        result = []
        for path in paths:
            assert path.endswith('.jsonl')
            result.extend(read_json(path))
        return result

def read_json(path, generator_mode=False, force_type=None):
    path = Path(path)
    if path.suffix == '.json' or force_type == 'json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif path.suffix == '.jsonl' or force_type == 'jsonl':
        if generator_mode:
            data = _read_json_generator(path)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    tmp = smart_json_loads(line)
                    if tmp:
                        data.append(tmp)
                
    else:
        raise ValueError(f'Unsupported file format: {path.suffix}')
    return data


def custom_serializer(obj):
    try:
        return json.JSONEncoder().default(obj)
    except TypeError:
        warnings.warn(f"{str(obj)} is not serializable, we use str() to prevent exception!")
        return str(obj)

def write_json(data, path, create_parent_dir=True, count=None, enable_tqdm=False, open_kwargs=None):
    path = Path(path)
    if not path.parent.exists() and create_parent_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, default=custom_serializer)
    elif path.suffix == '.jsonl':
        # configurate tqdm params
        if enable_tqdm:
            if inspect.isgenerator(data):
                if count is None:
                    warnings.warn('You should pass count')
                    enable_tqdm=False
            else:
                count = len(data)
        open_kwargs_ = {'encoding': 'utf-8', }
        if open_kwargs is not None:
            open_kwargs_.update(open_kwargs)
        
        with open(path, 'w', **open_kwargs_) as f:
            if enable_tqdm:
                pbar = tqdm(total=count)
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False, default=custom_serializer) + '\n')
                if enable_tqdm:
                    pbar.update(1)
    else:
        raise ValueError(f'Unsupported file format: {path.suffix}')

class JsonWrap():
    def __init__(self, input_file, index_file=None, subset_lambda=None, skip_error=True):
        assert input_file is not None
        self.input_file = input_file
        self.skip_error = skip_error
        if index_file is None or index_file == 'auto':
            # 对于jsonl 自动搜索idx
            input_file = Path(input_file)
            if input_file.suffix == '.jsonl':
                may_index_file = Path(input_file.parent, input_file.stem+'.idx')
                if may_index_file.exists():
                    index_file = may_index_file
            if index_file == 'auto':
                index_file = None
        else:
            index_file = str(index_file)
        if index_file is None or len(str(index_file))==0:
            warnings.warn(f'Suggesting to add a index_file for {input_file}')
            self.data = read_json(input_file)
            self.mode = 'slow'
        else:
            assert Path(input_file).name.endswith('.jsonl')
            with open(index_file, 'rb') as f:
                self.indices = pickle.load(f)
            with open(input_file, "r+b") as f:
                self.mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.mode = 'fast'

        if subset_lambda is not None:
            self.subset_lambda = eval(subset_lambda)
            if self.mode == 'slow':
                self.data = [x for di, x in enumerate(self.data) if self.subset_lambda(di, x)]
            else:
                # jsonl with index do not supports content based filter
                self.subset_indices = [di for di in range(len(self.indices)) if self.subset_lambda(di, None)]
        else:
            self.subset_lambda = lambda x: True
            self.subset_indices = None
            
    def __len__(self):
        if self.mode == 'slow':
            return len(self.data)

        if self.subset_indices is not None:
            return len(self.subset_indices)
        return len(self.indices)
    
    def __getitem__(self, idx):
        if self.mode == 'slow':
            json_obj = self.data[idx]
        else:
            if self.subset_indices is not None:
                idx = self.subset_indices[idx]

            start = self.indices[idx]
            end = self.indices[idx + 1] if idx + 1 < len(self.indices) else self.mmapped_file.size()
            json_line = self.mmapped_file[start:end].decode('utf-8')
            
            json_obj = smart_json_loads(json_line)
            if json_line is None and self.skip_error:
                return self.__getitem__(idx+1)
        return json_obj
    
    def close(self,):
        if hasattr(self, 'mmapped_file') and self.mmapped_file is not None:
            self.mmapped_file.close()
            self.mmapped_file = None

if __name__ == '__main__':
    d = read_json('/Users/luke/Documents/common/val.jsonl', generator_mode=True)
    print(len(list(d)))