import os
import time
import subprocess

import mss
import numpy as np
import pyautogui
import cv2
import pygetwindow as gw

from .BaseEnv import BaseEnv
from cv_matching import cv_matching

class ThetanArenaEnv(BaseEnv):
    def __init__(self, io_mode=BaseEnv.IO_MODE.FULL_CONTROL,
                 explore_space=BaseEnv.EXPLORE_MODE.FULL):
        """This is the code of start game

        Use try and catch statement to catch exception
        when the game is not installed in the provided path,
        then raise the exception for the upper level to handle.

        The hardcode path to open the game:
        "C:\Program Files (x86)\Thetan Arena\Thetan Arena.exe"
        """
        super(ThetanArenaEnv, self).__init__()

        try:
            self._start_game()
        except:
            raise Exception("the game is not installed")
        
        if io_mode == IO_MODE.FULL_CONTROL:
            # obs_shape in (HEIGHT, WIDTH, N_CHANNELS)
            obs_shape = (512, 512, 4)
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=obs_shape,
                                                dtype=np.uint8)

            gameWindow = gw.getWindowsWithTitle('Thetan Arena')[0]
            self.monitor = {"top": gameWindow.top,
                            "left": gameWindow.left,
                            "width": gameWindow.width,
                            "height": gameWindow.height}

            self.sct = mss.mss()
            img = np.array(self.sct.grab(self.monitor))
            ratio = max(img.shape) // 512
            self.dsize = (img.shape[1] // ratio,
                          img.shape[0] // ratio)
            self.left_right_pad = (obs_shape[1] - self.dsize[1]) // 2
            self.top_bottom_pad = (obs_shape[0] - self.dsize[0]) // 2
            
        self.info = {'waiting': True}
        self.done = False
        self.rewards = 0

        return self.action_space, self.observation_space

    def step(self, action):
        self._take_action(action)
        obs = self._screen_cap()
        self._check_if_game_session_started()
        self._check_if_game_session_ended()
        return obs, self.rewards, self.done, self.info

    def reset(self):
        self._reset_game()
        self.enter_match()
        self.info = {'waiting': True}

    def close(self):
        self._end_game()

    def _take_action(self, action):
        self._mouse_move(action[0,-5:-3])
        self._mouse_press(action[0,-3:-1])
        self._keyboard_press(action[0,:-5])
        self._mouse_move(action[1,-5:-3])
        self._mouse_release(action[1,-3:-1])
        self._keyboard_release(action[1:-5])

    def _screen_cap(self):
        """This is the function to capture screen from the game Thetan Arena.

        `pygetwindow` is used to get the coordinates and resolution of game
        by matching with WindowsWithTitle('Thetan Arena').
        
        The image will then be resized to the specified size defined in 
        `observation_space` with the same ratio as original image by padding
        with black pixels.

        Returns
        -------
        numpy.array
            Captured RGBA image with format of numpy.array in HWC
        
        """
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.resize(img, self.dsize)
        return cv2.copyMakeBorder(img,
                                  self.top_bottom_pad,
                                  self.top_bottom_pad,
                                  self.left_right_pad,
                                  self.left_right_pad,
                                  cv2.BORDER_CONSTANT,
                                  value=[0, 0, 0])

    def _keyboard_input(self, action):
        # scan and find keyboard action
        # key press # self._keyboard_press(ascii_list)
        # key release # self._keyboard_release(ascii_list)
        pass

    def _keyboard_press(self, ascii_list):
        pass

    def _keyboard_release(self, ascii_list):
        pass
    
    def _mouse_move(self, width, height):
        """Move mouse to the location of the percentage of game window
        width and height.

        Parameters
        ----------
        width : float
          percentage of screen width
        height : float
          percentage of screen height
        """
        x = (self.monitor['left'] +
             width *
             abs(self.monitor['right'] -
                 self.monitor['left']))
        y = (self.monitor['top'] +
             height *
             abs(self.monitor['bottom'] -
                 self.monitor['top']))
        pyautogui.moveTo(x, y, duration=0.2)
    
    def _mouse_click(self, action):
        """Click mouse left and right button by probability value.

        Parameters
        ----------
        action : numpy.array
          `numpy.array` with shape of `(2, 2)` contain the probabilities of
          left and right mouse down/up, trigger left mouse down/up if value
          of the field greater than 0
          ```
          # definition of teh fields
          [
            [left_mouse_down, right_mouse_down],
            [left_mouse_up, right_mouse_up]
          ]
          ```
        """
        # mouse press
        self._mouse_press(*(action[0] > 0))
        # mouse release
        self._mouse_release(*(action[1] > 0))

    def _mouse_press(self, left, right):
        """Press and hold mouse left and right button by boolean.

        Parameters
        ----------
        left : bool
          flag for left click, trigger left click if true
        right : bool
          flag for right click, trigger right click if true
        """
        if left:
            pyautogui.mouseDown()
        if right:
            pyautogui.mouseDown(button='right')

    def _mouse_release(self, left, right):
        """Release mouse left and right button by boolean.

        Parameters
        ----------
        left : bool
          flag for releasing left click, trigger release if true
        right : bool
          flag for releasing right click, trigger release if true
        """
        if left:
            pyautogui.mouseUp()
        if right:
            pyautogui.mouseUp(button='right')

    def _start_game(self):
        """This is the code for starting game

        The file path "C:\\Program Files (x86)\\Thetan Arena\\Thetan Arena.exe"
        and
        program name "C:\\Program Files (x86)\\Thetan Arena\\Thetan Arena.exe"
        is hardcode.
        """

        progname = "C:\\Users\\Public\\Desktop\\Thetan Arena"
        filepath = "C:\\Program Files (x86)\\Thetan Arena\\Thetan Arena.exe"

        self.p = subprocess.Popen([filepath, progname])

    def enter_match(self):
        """
        change the current directory to the script folder so that relative
        path can be used
        """
        folder_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(folder_path)

        """
        opencv read the find match image by file path
        """

        tofind = cv2.imread(
            "../SourcePictures/findmatch2.png", cv2.IMREAD_UNCHANGED)

        """
        using the above method
        """
        loc, thershold = cv_matching.matching_with_screencap(tofind)

        """
        using pyautogui to click the obtained coordinates
        """
        pyautogui.moveTo(loc[0] - 100, loc[1], duration=0.2)
        pyautogui.click(button="left", duration=0.2)

        """
        similarly, using pyautogui to click the obtained coordinates
        """

        tofind = cv2.imread(
            "../SourcePictures/deathmatch2.png", cv2.IMREAD_UNCHANGED)
        loc2, thershold = cv_matching.matching_with_screencap(tofind)
        if thershold > 0.7:
            pyautogui.moveTo(loc2[0] + 250, loc2[1], duration=0.2)
            pyautogui.click(button="left", duration=0.2)

        else:
            pyautogui.dragTo(loc[0] - 400, loc[1], duration=0.2, button="left")
            tofind = cv2.imread(
                "../SourcePictures/deathmatch2.png",
                cv2.IMREAD_UNCHANGED)
            time.sleep(2)
            loc2, thershold = cv_matching.matching_with_screencap(tofind)
        if thershold < 0.6:
            print("no deathmatch game mode")
            exit()
        pyautogui.moveTo(loc2[0] + 250, loc2[1], duration=0.2)
        pyautogui.click(button="left", duration=0.2)

        """
        again, using the above method to find and click the tutorial image
        """

        tofind = cv2.imread("../SourcePictures/tutor.png", cv2.IMREAD_UNCHANGED)
        loc, thershold = matching_with_screencap(tofind)
        pyautogui.moveTo(loc[0], loc[1], duration=0.2)
        pyautogui.click(button="left", duration=0.2)

    def _end_game(self):
        """
        This is the code for end game
        """
        self.p.terminate()
        
    def _check_if_game_session_started(self):
        """Determine if the Game session has started
        
        Powered by OpenCV template matching of tutorial session start screen
        """
        # opencv read the image by file path for templatematching
        tofind = cv2.imread(
            "../SourcePictures/entertutor2.png", cv2.IMREAD_UNCHANGED)
        loc, thershold = cv_matching.matching_with_screencap(tofind)
        if thershold > 0.7:
            self.info['waiting'] = False

    def _check_if_game_session_ended(self):
        # self.rewards = self._screen_get_total_score()
        """Determine if the Tutorial has finished

        Powered by OpenCV template matching
        """
        if not self.info['waiting']:
            # opencv read the image by file path for template matching
            tofind = cv2.imread(
                "../SourcePictures/finishtutor.png",
                cv2.IMREAD_UNCHANGED)

            loc, thershold = matching_with_screencap(tofind)
            if thershold > 0.7:
                self.done = True

    def _reset_game(self):
        pass
