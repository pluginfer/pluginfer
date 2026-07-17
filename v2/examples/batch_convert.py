
def convert(filename, outname):
    try:
        with open(filename, 'r', encoding='utf-16') as f:
            content = f.read()
    except:
        try:
            with open(filename, 'r', encoding='utf-16-le') as f:
                content = f.read()
        except:
             with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
    with open(outname, 'w', encoding='utf-8') as f:
        f.write(content)

convert('final_functional_results.txt', 'final_functional_results_utf8.txt')
convert('final_security_results.txt', 'final_security_results_utf8.txt')
