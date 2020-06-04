#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import cv2
import os
import scenedetect
import pandas as pd


# In[2]:


import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.detectors import ThresholdDetector
from scenedetect.scene_detector import SparseSceneDetector


# In[3]:


#Example code from scenedetect
from __future__ import print_function
import os

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

STATS_FILE_PATH = 'testvideo.stats.csv'

def main():

    # Create a video_manager point to video file testvideo.mp4. Note that multiple
    # videos can be appended by simply specifying more file paths in the list
    # passed to the VideoManager constructor. Note that appending multiple videos
    # requires that they all have the same frame size, and optionally, framerate.
    video_manager = VideoManager(['testvideo.mp4'])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector(threshold=10))
    base_timecode = video_manager.get_base_timecode()

    try:
        # If stats file exists, load it.
        if os.path.exists(STATS_FILE_PATH):
            # Read stats from CSV file opened in read mode:
            with open(STATS_FILE_PATH, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        #start_time = base_timecode + 20     # 00:00:00.667
        #end_time = base_timecode + 10000.0     # 00:00:20.000
        # Set video_manager duration to read frames from 00:00:00 to 00:00:20.
        #video_manager.set_duration(start_time=start_time, end_time=end_time)

        # Set downscale factor to improve processing speed (no args means default).
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Like FrameTimecodes, each scene in the scene_list can be sorted if the
        # list of scenes becomes unsorted.

        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i+1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(STATS_FILE_PATH, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()


# In[4]:


main()


# In[5]:


vidDirectory = 'D:/Interactive Video Retrieval/videos/'

vidID = os.listdir(vidDirectory)


# In[19]:


for ID in vidID:
    
    print(ID)
    
    IDdir = os.listdir(vidDirectory+ID+'/')
    
    #if(IDdir.count(ID+'scenes.csv')):
        #print(ID)
        #continue
    
    if(IDdir.count(ID+'.mov')):
        video_manager = VideoManager([vidDirectory+ID+'/'+ID+'.mov'])
    elif(IDdir.count(ID+'.m4v')):
        video_manager = VideoManager([vidDirectory+ID+'/'+ID+'.m4v'])
    else:
        video_manager = VideoManager([vidDirectory+ID+'/'+ID+'.mp4'])
        
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    try:
        # If stats file exists, load it.
        if os.path.exists(vidDirectory+ID+'/'+ID+'stats.csv'):
            # Read stats from CSV file opened in read mode:
            with open(vidDirectory+ID+'/'+ID+'stats.csv', 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed (no args means default).
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Like FrameTimecodes, each scene in the scene_list can be sorted if the
        # list of scenes becomes unsorted.
        
        #Save the intervals
        numpy.savetxt(vidDirectory+ID+'/'+ID+'scenes.csv', [a for a in scene_list], delimiter=",")

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(vidDirectory+ID+'/'+ID+'stats.csv', 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()

