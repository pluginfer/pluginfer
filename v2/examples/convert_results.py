
try:
    with open('security_results.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except:
    with open('security_results.txt', 'r', encoding='utf-16-le') as f:
        content = f.read()

with open('security_results_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
