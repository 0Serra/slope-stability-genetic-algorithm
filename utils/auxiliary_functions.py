

def progress_bar(current_value, total_value):
    percent = (current_value / total_value) * 100
    progress = int(percent)
    bar = '/' * progress + '-' * (100 - progress)
    print(f"\r{bar} [ {percent:.2f} % ]", end="")
