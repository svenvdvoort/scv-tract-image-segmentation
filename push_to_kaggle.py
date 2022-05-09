""" Python script that automates running the notebook and included scripts on Kaggle """
import re
import subprocess
import time

try:
    kaggle_config_output = subprocess.check_output("kaggle config view", shell=True).decode()
    username = re.search("username: .*\n", kaggle_config_output).group(0).split()[1]
except Exception as e:
    print(f"Could not get username from kaggle command: {e}")
    exit()

# Subsitute INSERT_USERNAME with actual Kaggle username in metadata files
with open("notebook_metadata/kernel-metadata.json", "r") as f:
    notebook_metadata = f.read()
with open("utility_metadata/kernel-metadata.json", "r") as f:
    utility_metadata = f.read()
with open("notebook_metadata/kernel-metadata.json", "w") as f:
    f.write(notebook_metadata.replace("INSERT_USERNAME", username))
with open("utility_metadata/kernel-metadata.json", "w") as f:
    f.write(utility_metadata.replace("INSERT_USERNAME", username))

subprocess.run("kaggle kernels push -p ./utility_metadata/", shell=True)
print("SCV utility script was pushed. Now waiting 30 sec for it to become available...")
for _ in range(30):
    time.sleep(1)
    print(".", end="", flush=True)
print()
subprocess.run("kaggle kernels push -p ./notebook_metadata", shell=True)

# Revert to original metadata files
with open("notebook_metadata/kernel-metadata.json", "w") as f:
    f.write(notebook_metadata)
with open("utility_metadata/kernel-metadata.json", "w") as f:
    f.write(utility_metadata)

print("Push to Kaggle complete")
