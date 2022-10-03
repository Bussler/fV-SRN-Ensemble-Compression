import os.path
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import common.utils
import pyrenderer
from inference.volume import RenderTool

FILE = 'datasets/Ejecta/snapshot_070_256.cvol'
#FILE = 'experimentRuns/parameter_normalization/m01.cvol'
RENDERCONFIGFILE = 'config-files/ejecta70-v6-dvr.json'

torch.set_num_threads(12)
device = torch.device('cuda:0')

def loadVolume(file_name):
    if not os.path.exists(file_name):
        raise ValueError(f'[ERROR] Volume file {file_name} does not exist.')
    vol = pyrenderer.Volume(file_name)
    return vol

def _render_and_refine(image_evaluator):
    resolution = (512, 512)
    img = image_evaluator.render(*resolution)
    # tonemapping
    img = image_evaluator.extract_color(img)
    return img

def renderCVol():
    args = {'renderer:settings_file': RENDERCONFIGFILE}

    vol = loadVolume(FILE)
    render_tool = RenderTool.from_dict(args, device, )
    image_evaluator = render_tool.set_source(vol).get_image_evaluator()
    image_evaluator.camera.pitchYawDistance.value = render_tool.default_camera_pitch_yaw_distance()
    img = _render_and_refine(image_evaluator).cpu().numpy()

    img = np.squeeze(img)
    img = np.resize(img, (img.shape[1], img.shape[2], img.shape[0]))
    img = (img * 255).astype(np.uint8)

    printImg = Image.fromarray(img) #Image.fromarray(img, 'RGB')
    imageName = 'testImage.png'
    printImg.save(imageName)
    print("Rendering Complete. Saved to: ", imageName)


if __name__== '__main__':
    renderCVol()