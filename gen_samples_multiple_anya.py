import os
import tempfile
import subprocess


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as working_dir:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        outdir = "out_knee"
        trunc = "0.7"
        seeds = "0-1"
        fov_deg = "45"
        network = "./training-runs/00001-drr-knee_128_cx-gpus1-batch8-gamma5/network-snapshot-006000.pkl"

        # Define the parameters
        angle_p = [0, 0.5, 1, 1.57, 2, 2.5, 3, -0.5, -1, -1.57, -2, -2.5, -3]
        angle_y = [0.4, 0.8, 1.2]
        cam_radius = [5, 100, 500, 1000, 1500]

        for y in angle_y:
            print(f"Running for yaw {y}...")
            for radius in cam_radius:
                for p in angle_p:
                    # Command and arguments
                    cmd = f"python {os.path.join(dir_path, 'gen_samples.py')}"
                    cmd += f" --outdir {outdir} --trunc {trunc} --seeds {seeds} --fov-deg {fov_deg} --network {network} --pitch {p} --yaw {y} --cr {radius}"
                    # Construct the full command
                    subprocess.run([cmd], shell=True)



