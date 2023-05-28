import os
import json
import random
def read_directory(path, result):
    paths = os.listdir(path)
    for i, item in enumerate(paths) :
        sub_path = os. path. join(path, item)
        if os.path.isdir(sub_path):
            result[item] = {}
            read_directory(sub_path, result[item])
        else:
            result[item] = item

if __name__ == "__main__":
    fpath = '/root/autodl-tmp/code/dataset_13'
    filename = '/root/autodl-tmp/code/results/json_13.txt'
    result = {}
    output = []
    read_directory( fpath, result)
    for item in result:
        sub_path = fpath + "/" + item
        if os.path.isdir(sub_path):
            files = os.listdir(sub_path)
            if item != "case_00210":
                output.append({
                    "image": item + "/imaging.nii.gz",
                    "label": item + "/segmentation.nii.gz"
                })
            else:
                break
    random.shuffle(output)
    print(output)
    json_res = json.dumps(output, indent=2)
    with open(filename, 'w' ) as fp:
        fp.write(json_res)