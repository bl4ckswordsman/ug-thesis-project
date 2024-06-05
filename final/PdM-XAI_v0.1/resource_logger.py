import psutil


def log_cpu_usage():
    cpu_usage = psutil.cpu_percent()
    return cpu_usage
