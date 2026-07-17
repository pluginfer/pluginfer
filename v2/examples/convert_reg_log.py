
try:
    with open('full_regression_log.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except:
    try:
         with open('full_regression_log.txt', 'r', encoding='utf-16-le') as f:
            content = f.read()
    except:
         with open('full_regression_log.txt', 'r', encoding='utf-8') as f:
            content = f.read()

with open('full_regression_log_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
