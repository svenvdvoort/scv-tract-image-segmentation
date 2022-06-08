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
with open("eval_notebook_metadata/kernel-metadata.json", "r") as f:
    eval_notebook_metadata = f.read()
with open("utility_metadata/kernel-metadata.json", "r") as f:
    utility_metadata = f.read()
with open("model_data/dataset-metadata.json", "r") as f:
    model_data_metadata = f.read()
with open("notebook_metadata/kernel-metadata.json", "w") as f:
    f.write(eval_notebook_metadata.replace("INSERT_USERNAME", username))
with open("utility_metadata/kernel-metadata.json", "w") as f:
    f.write(utility_metadata.replace("INSERT_USERNAME", username))
with open("model_data/dataset-metadata.json", "w") as f:
    f.write(model_data_metadata.replace("INSERT_USERNAME", username))

subprocess.run("kaggle kernels push -p ./utility_metadata/", shell=True)
print("SCV utility script was pushed. Now waiting 30 sec for it to become available...")
for _ in range(30):
    time.sleep(1)
    print(".", end="", flush=True)
print()
subprocess.run("kaggle datasets create -p ./model_data/", shell=True)
print("Model data was updated. Now waiting 30 sec for it to become available...")
for _ in range(30):
    time.sleep(1)
    print(".", end="", flush=True)
print()
subprocess.run("kaggle kernels push -p ./eval_notebook_metadata", shell=True)

# Revert to original metadata files
with open("notebook_metadata/kernel-metadata.json", "w") as f:
    f.write(eval_notebook_metadata)
with open("utility_metadata/kernel-metadata.json", "w") as f:
    f.write(utility_metadata)
with open("model_data/dataset-metadata.json", "w") as f:
    f.write(model_data_metadata)

print("Push to Kaggle complete")
