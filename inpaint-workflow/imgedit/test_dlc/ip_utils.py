import netifaces
import socket

def get_local_ip():
    """获取本地 IP 地址，优先返回非回环地址"""
    # 方法1: 尝试通过连接外部地址获取本地 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # 方法2: 使用 netifaces 查找
    try:
        interfaces = netifaces.interfaces()
        # 优先查找以 10. 开头的 IP（内网地址）
        for interface in interfaces:
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                for addr_info in addresses[netifaces.AF_INET]:
                    ip_address = addr_info['addr']
                    if ip_address.startswith('10.') or ip_address.startswith('192.168.') or ip_address.startswith('172.'):
                        return ip_address

        # 如果没有找到内网地址，返回第一个非回环地址
        for interface in interfaces:
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                for addr_info in addresses[netifaces.AF_INET]:
                    ip_address = addr_info['addr']
                    if not ip_address.startswith("127."):
                        return ip_address
    except Exception:
        pass

    # 方法3: 返回 localhost
    return "localhost"