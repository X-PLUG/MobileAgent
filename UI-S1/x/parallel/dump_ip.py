

from pathlib import Path
import warnings
import click

import socket
import random

from x.utils.network import get_local_ip

def find_available_port():
    while True:
        # 随机选择一个端口号，范围通常在1024到65535
        port = random.randint(1024, 65535)
        
        # 尝试绑定到该端口，以检查其可用性
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue

def load_ips(exp, path="dist_log"):
    ips = [path for path in Path(path, exp).iterdir()]
    ips = [{'ip': _.name.split(':')[0], 'port': _.name.split(':')[1], 'raw_path': _} if ':' in _.name else {'ip': _.name.split(':')[0], 'raw_path': _} for _ in ips]
    return ips
    
def delete_ips(ips, exp, path="/dist_log"):
    pass

@click.command()
@click.option('--exp', type=str)
@click.option('--path', default="dist_log", type=str)
@click.option('--ip', default='', type=str)
@click.option('--port', default='', type=str)
def cli(exp, path, ip, port):
    import socket
    if len(ip) == 0:
        ip = get_local_ip(interface='eth0')
    if len(port) == 0:
        port = find_available_port()
    
    print(port)
    Path(path, exp).mkdir(exist_ok=True, parents=True)

    open(Path(path, exp, f"{ip}:{port}"),'w').close()

if __name__ == '__main__':
    cli()