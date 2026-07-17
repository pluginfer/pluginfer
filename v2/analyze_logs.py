
import re

LOG_FILE = "node_final_test.log"

try:
    with open(LOG_FILE, "r", encoding="utf-16") as f:
        content = f.read()
except:
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read log: {e}")
        exit()

# Search for plugin load messages
loaded = re.findall(r"Loaded plugin: (\w+)", content)
failed = re.findall(r"Failed to load .*?: (.*)", content)
errors = re.findall(r"Error.*", content, re.IGNORECASE)

print(f"Loaded Plugins ({len(loaded)}): {loaded}")
print(f"Failed Plugins ({len(failed)}): {failed}")
print("\n--- Errors ---")
for e in errors[:20]:
    print(e)
