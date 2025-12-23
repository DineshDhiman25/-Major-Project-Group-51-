import os

folder = "/Users/aditsg/Library/Application Support/PrismLauncher/instances/Prominence II Hasturian Era /minecraft/mods"

jar_files = [f for f in os.listdir(folder) if f.endswith(".jar")]
print(jar_files)
