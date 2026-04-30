import os

def resolve_file(filepath):
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("<<<<<<<") or line.startswith("=======") or line.startswith(">>>>>>>"):
                continue
            lines.append(line)
    
    with open(filepath, "w") as f:
        f.writelines(lines)

resolve_file("esp_data/transforms/__init__.py")
resolve_file("esp_data/datasets/__init__.py")
