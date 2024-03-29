from functools import wraps
from collections import abc
import numpy as np
from tifffile import TiffWriter
import cv2

def coroutine(func):
    """Decorator for priming a coroutine (func)"""
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return primer

def save_avg_profile(profile, path, name):
        profile = np.array(profile)
        profile_mean = np.mean(profile, axis=0)
        X = np.arange(0,profile_mean.shape[0],1)
        to_save = np.array([X,profile_mean]).T
        np.savetxt(path + "\\"+name+".txt", to_save)

@coroutine
def z_stack_microservice(path):
    results = {"width":[], "intensity":[]}
    while True:
        data = yield
        if data is None:
            break
        else:
            results["width"].append(data[0])
            results["intensity"].append(data[1])
    width = np.array(results["width"])
    intensity = np.array(results["intensity"])
    relative = intensity/width
    whole_data = np.zeros((width.shape[0],3))
    whole_data[:,0] = width
    whole_data[:,1] = intensity
    whole_data[:,2] = relative
    np.savetxt(path + "_" + "evaluation" + ".txt", whole_data)

@coroutine
def mic_project_generator(path, i):
    profiles = []
    j = -1
    while True:
        data = yield
        if data is None:
            break
        else:
            profile,z = data
            if z==0:
                profiles.append([])
                j += 1
        profiles[j].append(profile)
    for j,profile in enumerate(profiles):
        profile = np.array(profile)
        profiles[j] = np.vstack(profile)
    profiles = np.array(profiles)
    out = np.mean(profiles, axis=0)
    mic_project = out * 100
    with TiffWriter(path + r"\mic_project" + str(i) + ".tif") as tif:
        tif.save(mic_project.astype(np.uint16), photometric='minisblack')
    return out

@coroutine
def profile_collector(path, i):
    profiles = {"red":[],"green":[],"blue":[]}
    while True:
        profile = yield
        if profile is None:
            break
        if isinstance(profile, abc.Sequence):
            profiles["red"].append(profile[0])
            profiles["green"].append(profile[1])
            try:
                profiles["blue"].append(profile[2])
            except IndexError:
                print("tuple value out of range")
        else:
            profiles["red"].append(profile)
    for key,item in profiles.items():
        if item:
            save_avg_profile(item, path, key+str(i))
    return profiles


@coroutine
def profile_painter(image, path):
    image = image.astype(np.uint16)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    if len(image.shape) !=3:
        raise ValueError(f"Image should be RGBA found {image.shape}")
    profiles = np.zeros_like(image).astype(np.uint16)
    while True:
        data = yield
        if data is None:
            break
        if isinstance(data, abc.Sequence):
            line,color = data
        else:
            line = data
            color = (1.0,0.0,0.0,1.0)
        if line['X'].min() >= 0 and line['Y'].min() >= 0 and line['X'].max() < image.shape[0] and line['Y'].max() < image.shape[1]:
            profiles[line['X'].astype(np.int32), line['Y'].astype(np.int32)] = np.array(color) * 50000
    image_writer(image, profiles, path)

@coroutine
def profile_painter_2(image, path):
    image = image.astype(np.uint16)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    if len(image.shape) !=3:
        raise ValueError(f"Image should be RGBA found {image.shape}")
    profiles = np.zeros_like(image).astype(np.uint16)
    while True:
        data = yield
        if data is None:
            break
        if isinstance(data, abc.Sequence):
            line,color = data
        else:
            line = data
            color = (1.0,0.0,0.0,1.0)
        if line.x.min() >= 0 and line.y.min() >= 0 and line.x.max() < image.shape[0] and line.y.max() < image.shape[1]:
            profiles[line.x.astype(np.int32), line.y.astype(np.int32)] = np.array(color) * 50000
    image_writer(image, profiles, path)


def image_writer(image, profiles, path):
    with TiffWriter(path + r'\Image_with_profiles.tif') as tif:
        tif.save(np.asarray(profiles[...,0:3]).astype(np.uint16), photometric='rgb')
    image = image.astype(np.uint32)*10
    image += profiles
    image = np.clip(image, 0, 65535)
    with TiffWriter(path + r'\Image_overlay.tif') as tif:
        tif.save(image[...,0:3].astype(np.uint16), photometric='rgb')