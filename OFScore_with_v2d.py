import subprocess
import argparse
import os
import json
import tarfile

# python trigger.py --input_folder="/path/to/mp4/folder" 

parser = argparse.ArgumentParser ()

#parser.add_argument ("--task", default = "idle", help = "the task you want to run")
parser.add_argument ("--input_folder", default = "./", help = "path to the folder that contains .mp4 files")
parser.add_argument ("--tmp_folder", default = "optical_flow_temporary_folder", help = "folder where v2d places its output")
parser.add_argument ("--output_folder", default = "./", help = "path where you want to place the wds")
parser.add_argument ("--output_filename", default = "OFresult")

args = parser.parse_args ()

#if args.task == "oflow":
if True:
	input_folder = args.input_folder
	# convert to 'webdataset'
	with tarfile.open ("00000.tar", "w") as tar:
		for i in os.listdir (input_folder):
			if i.endswith (".mp4"):
				file_name = os.path.splitext (i)[0]
				json_content = {"key": file_name, "error_message": None}
				i = os.path.join (input_folder, i)
				j = os.path.join (input_folder, f"{file_name}.json")
				with open (j, "w") as json_file:
					json.dump (json_content, json_file)
				tar.add (i)
				tar.add (j)
	for i in os.listdir (input_folder):
		if i.endswith (".mp4"):
			continue
		else:
			i = os.path.join (input_folder, i)
			subprocess.run (["rm", i])
	
	# run v2d
	tmp_folder = args.tmp_folder
	subprocess.run ([
    "video2dataset",
    "--url_list=./00000.tar",
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
	output_folder = args.output_folder
	subprocess.run (["mkdir", output_folder])
	ofres = []
	for R, D, F in os.walk (tmp_folder):
		for d in D:
			dr = os.path.join (tmp_folder, d)
			for i in os.listdir (dr):
				if i.endswith ("json"):
					i = os.path.join (dr, i)
					with open (i, "r") as single_of:
						k = json.load (single_of)
					ofres.append ({"key": k.get ("key"), "meanOF": k.get ("mean_optical_flow_magnitude")})
	subprocess.run (["rm", "-rf", tmp_folder])
	output = os.path.join (output_folder, f"{args.output_filename}.json")
	with open (output, "w") as j:
		json.dump (ofres, j, indent = 1)
