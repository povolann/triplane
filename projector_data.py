import zipfile
import json
import os
import click
import numpy as np

# outdir = '/home/anya/Programs/EG3D-projector/eg3d/projector_test_data/'
# name = '000001'
# focal_length = 1.7074
# intrinsics = np.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]]).reshape(-1)
# pose_path = '/home/anya/Programs/eg3d/dataset_preprocessing/shapenet_cars/cars_train/1a1dcd236a1e6133860800e6696b8284/pose/000001.txt'

# with open(pose_path, 'r') as f:
#     pose = np.array([float(n) for n in f.read().split(' ')])

# c = np.concatenate([pose, intrinsics]) # c.shape = (25,)
# np.save(os.path.join(outdir, f'{name}.npy'), c)


def read_json_from_zip(zip_path, json_file_name):
    # Check if the zip file exists
    if not os.path.isfile(zip_path):
        return "Zip file does not exist."

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Check if the json file exists in the zip
        if json_file_name in z.namelist():
            # Extract the JSON file
            with z.open(json_file_name) as f:
                # Read the JSON file data
                json_data = json.load(f)
                return json_data
        else:
            return f"JSON file {json_file_name} does not exist in the zip archive."

@click.command()
@click.option('--data', 'data', help='Data path', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--num_img', 'num_img', type=int, help='Which image to take out', default=5, show_default=True)
def run(
        data: str,
        outdir: str,
        num_img: int
):
    """Get image and coordinate matrix from zip file used for EG3D training.
    Examples:
    python projector_data.py --data=/home/anya/Programs/eg3d/eg3d/datasets/cars_128.zip --outdir=./projector_test_data

    """

    os.makedirs(outdir, exist_ok=True)

    print('Loading data from from "%s"...' % data)

    json_file_name = 'dataset.json'
    json_data = read_json_from_zip(data, json_file_name)
    
    # Choose image and c
    image_path = json_data['labels'][num_img][0]
    image_name = os.path.basename(image_path)
    image_out = os.path.join(outdir, image_name)

    c = json_data['labels'][num_img][1]
    c = np.array(c)
    print(c)
    np.save(os.path.join(outdir, f'{image_name[:-4]}.npy'), c)


    # Copy image to the output directory
    with zipfile.ZipFile(data, 'r') as z:
        # Extract only the specified image to the destination directory
        for file in z.namelist():
            if file == image_path:
                z.extract(file, outdir)
                print(f"Image '{image_path}' has been extracted to '{image_out}'")
                break
        else:
            print(f"Image '{image_path}' not found in the zip file.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
     run() 

# ----------------------------------------------------------------------------