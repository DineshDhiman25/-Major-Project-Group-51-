import argostranslate.package, argostranslate.translate
from pathlib import Path

# Download and install the package index (list of all models)
argostranslate.package.update_package_index()

# Get all available translation packages
available_packages = argostranslate.package.get_available_packages()

# Define pairs you want
target_pairs = [
    ("ru", "en"),  # Russian → English
    ("en", "ru"),  # English → Russian
    ("uk", "en"),  # Ukrainian → English
    ("en", "uk")   # English → Ukrainian
]

# Filter only those from the list
selected_packages = [
    pkg for pkg in available_packages
    if (pkg.from_code, pkg.to_code) in target_pairs
]

# Download and install selected models
for pkg in selected_packages:
    print(f"Downloading: {pkg.from_code} → {pkg.to_code}")
    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)

print("Installed language models successfully!")
