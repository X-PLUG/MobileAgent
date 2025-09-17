import random
import string
from x.io import write_json


class FileBackendLoadBalanceParent():
    def __init__(self, quota: int) -> None:
        self.hash_key = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(20))
        self.quota = [0 for i in range(quota)]
        write_json(self.quota, f'/dev/shm/{self.hash_key}.json')
        