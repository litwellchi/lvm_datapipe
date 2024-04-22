import subprocess
import argparse
import os
import json
import tarfile

# python trigger.py --input_folder="/path/to/mp4/folder" 

parser = argparse.ArgumentParser()

#parser.add_argument("--task", default = "idle", help = "the task you want to run")
parser.add_argument("--input_folder", default = "./", help = "path to the folder that contains .mp4 files")
parser.add_argument("--output_folder", default = "./", help = "path where you want to place the wds")

args = parser.parse_args()
input_folder = args.input_folder
input_name = os.path.basename(input_folder)
tar_path = os.path.join(input_folder, '1.tar')
tmp_folder = os.path.join(args.output_folder, input_name+'_tmp')
output_folder = args.output_folder
os.makedirs(tmp_folder, exist_ok=True)


#if args.task == "oflow":
if True:
	# convert to 'webdataset'
	with tarfile.open(tar_path, "w") as tar:
		for i in os.listdir(input_folder):
			if i.endswith(".mp4"):
				file_name = os.path.splitext(i)[0]
				json_content = {"key": file_name, "error_message": None}
				i = os.path.join(input_folder, i)
				j = os.path.join(input_folder, f"{file_name}.json")
				with open(j, "w") as json_file:
					json.dump(json_content, json_file)
				tar.add(i)
				tar.add(j)
				subprocess.run(["rm", j])
	# run v2d
	print(tar_path,tmp_folder)
	subprocess.run([
    "video2dataset",
    f"--url_list={tar_path}",
    "--input_format=webdataset",
    "--output-format=files",
    f"--output_folder={tmp_folder}",
    "--stage",
    "optical_flow",
    "--encode_formats",
    '{"optical_flow": "npy"}',
    "--config",
    "optical_flow"])
	# combine result
	ofres = []
	for R, D, F in os.walk(tmp_folder):
		for d in D:
			dr = os.path.join(tmp_folder, d)
			for i in os.listdir(dr):
				if i.endswith("json"):
					i = os.path.join(dr, i)
					with open(i, "r") as single_of:
						k = json.load(single_of)
					ofres.append({"key": k.get("key"), "meanOF": k.get("mean_optical_flow_magnitude")})
	output = os.path.join(tmp_folder, f"OFresult.json")
	with open(output, "w") as j:
		json.dump(ofres, j, indent = 1)
	subprocess.run(["mv", output, input_folder])
	subprocess.run(["rm", "-rf", tmp_folder])
	subprocess.run(["rm", tar_path])