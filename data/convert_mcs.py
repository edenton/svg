import os
import glob
import argparse
from os import path
import subprocess
from tqdm import tqdm
import threading

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datadir", type=str, help="path to the mcs video dataset folder",
                    default="data/mcs_videos_1000")
parser.add_argument("-s", "--imsize", type=int, help="width of converted (square) images. default 64.", default=64)
args = parser.parse_args()

DATA_ROOT = args.datadir
IMSIZE = args.imsize

if not path.exists(DATA_ROOT):
    print(f'directory "{DATA_ROOT}" does not exist! Check arguments')
elif not path.exists(path.join(DATA_ROOT, 'raw')):
    print("Training videos must be in [datadir]/raw/(task)/*.mp4, where task is the task to which "
          "the video belongs")

def mp4_to_png_worker(path_to_task):
    task_name = path.basename(path_to_task[:-1])  # remove the '/' at the end of path to a folder
    vids = glob.glob(path.join(path_to_task, '*.mp4'))
    vids_tqdm = tqdm(sorted(vids))
    vids_tqdm.set_description(f'Task {task_name}')
    for vid in vids_tqdm:
        sample_name = path.basename(vid)
        sample_name = sample_name[:sample_name.rfind('.')]  # get rid of extension name
        out_folder = path.join(DATA_ROOT, 'processed', task_name, sample_name)
        os.makedirs(out_folder, exist_ok=True)
        ffmpeg_exe = 'ffmpeg'
        ffmpeg_args = f'-i "{vid}" -s {IMSIZE}x{IMSIZE} "{path.join(out_folder, sample_name + "_%04d.png")}"'

        try:
            subprocess.check_call(ffmpeg_exe + ' ' + ffmpeg_args, shell=True, stdout=subprocess.DEVNULL
                                  , stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print('convert_mcs.py: ffmpeg execution failed. Check that you have installed the ffmpeg executable'
                  'by typing "ffmpeg" in your terminal.')
            quit()


tasks = glob.glob(path.join(DATA_ROOT, 'raw', '*/'))
threads = []
for task in sorted(tasks):
    t = threading.Thread(target=mp4_to_png_worker, args=[task], daemon=True)
    t.start()
    threads.append(t)

for thread in threads:
    thread.join()