import time
from functools import wraps


__all__ = ["timer"]


def format_duration(duration):
    days = int(duration // 86400)
    hours = int((duration % 86400) // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60

    parts = []
    if days > 0:
        parts.append(f"{days} j")
    if hours > 0:
        parts.append(f"{hours} h")
    if minutes > 0:
        parts.append(f"{minutes} min")
    if seconds >= 1:
        parts.append(f"{seconds:.2f} s")
    elif seconds > 0:
        parts.append(f"{seconds*1000:.2f} ms")
    return ' '.join(parts)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"`{func.__name__}` execution time : {format_duration(end - begin)}")
        return result
    return wrapper



