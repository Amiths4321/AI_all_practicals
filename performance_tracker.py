import time


def start_timer():
    return time.perf_counter()


def elapsed_time(start_time):
    return round(time.perf_counter() - start_time, 4)