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
                 explore_space=BaseEnv.EXPLORE_MODE.MATCH):
        """Initialize environment, defines the input (action_space)
        and output (observation_space).
        
        Screen capture module shall be initialized here and gaming program
        shall be started. `observation_space` and `action_space` will be
        defined here, and `explore_space` will be defined here too.
        
        Parameters
        ----------
        io_mode : Enum, optional
            Default input/output of the environment interacting with reinforcement
            learning agent is `IO_MODE.FULL_CONTROL` which directly provide screen
            capture as `observation_space` for reinforcement learning agent to
            observe the state of environment and requires matrix of keys mapping to
            a 101 keys ANSI standard keyboard as keyboard input (allowing combined
            keys) plus a vector descripting `(X, Y)` mouse moving distance or mouse
            click event.
        explore_space : Enum, optional
            Explore space is the domain and scope of the tasks we want the
            reinforcement learning agent to learn and perform. Default explore space
            for the reinforcement learning agent is full control of the game from
            settings to character selection and weapon selection and joining the match.
            It is assumed that the reinforcement learning agent know what its going to
            do and do pick the right tools for its plan. Or we can set `explore_space`
            to `EXPLORE_MODE.MATCH` and write a script to settle everything else let
            the reinforcement learning agent focus on match in the game. 
            
        Returns
        -------
        action_space : gym.space
            A numpy.array compatible placeholder indicates the shape of input to `step()`
            for action instruction from the reinforcement learning agent (also as the
            output shape of the reinforcement learning agent).
        observation_space : gym.space
            A numpy.array compatible placeholder indicates the shape of output from `step()`
            for screen capture or data captured from game's Application Programming Interface
            (also as the input shape of the reinforcement learning agent).
        """
        super(ThetanArenaEnv, self).__init__()

        try:
            self._start_game()
        except:
            raise Exception("the game is not installed")
        
        if io_mode == IO_MODE.FULL_CONTROL:
            # press & release channels; 80 key + mouse (move + click + scroll)
            ACTION_SHAPE = (2, 80 + (2 + 2 + 1))
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=ACTION_SHAPE,
                                           dtype=np.float32)

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

            self.KEYBOARD_MAP = np.asarray([
                'altleft', 'altright', 'ctrlleft', 'ctrlright', 'shiftleft',
                'shiftright', 'backspace', 'capslock', 'delete', 'down',
                "'", ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
                '6', '7', '8', '9', ';', '=', '[', '\\', ']', '`', 'a',
                'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                'x', 'y', 'z', 'insert', 'left', 'num0', 'num1', 'num2',
                'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9',
                'end', 'enter', 'esc', 'numlock', 'pagedown', 'pageup',
                'right', 'space', 'tab', 'up', 'home'])
            
        self.info = {'waiting': True}
        self.done = False
        self.reward = 0

        self.cv_matcher = cv_matching()
        self.cv_matcher.preload_templates()

        return self.action_space, self.observation_space

    def step(self, action):
        self._take_action(action)
        obs = self._screen_cap()
        self._check_if_game_session_started()
        self._check_if_game_session_ended()
        return obs, self.reward, self.done, self.info

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

    def _capture_reward(self):
        if not self.info['waiting'] and not self.done:
            pass
        # self.reward = self._screen_get_total_score()

    def _keyboard_input(self, action):
        """Press and release keys based on `action` as KEYMAP mask

        It is allowed to mask a key to be pressed and not being
        released in the dim1 of `action` array for longer key press.

        Parameters
        ----------
        action : numpy.array
            Keymap mask of set of keys to be pressed in dim0
            and set of keys to be released in dim1
        """
        ascii_list = self.KEYBOARD_MAP[action[0] > 0]
        # key press
        self._keyboard_press(ascii_list)
        ascii_list = self.KEYBOARD_MAP[action[1] > 0]
        # key release
        self._keyboard_release(ascii_list)

    def _keyboard_press(self, ascii_list):
        for key in ascii_list:
            pyautogui.keyDown(key)

    def _keyboard_release(self, ascii_list):
        for key in ascii_list:
            pyautogui.keyUp(key)
    
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
        """This is the code of start game

        Use try and catch statement to catch exception
        when the game is not installed in the provided path,
        then raise the exception for the upper level to handle.

        The hardcode path to open the game:
        "C:\Program Files (x86)\Thetan Arena\Thetan Arena.exe"
        """
        progname = "C:\\Users\\Public\\Desktop\\Thetan Arena"
        filepath = "C:\\Program Files (x86)\\Thetan Arena\\Thetan Arena.exe"

        self.p = subprocess.Popen([filepath, progname])

    def enter_match(self):
        # using pyautogui to click the obtained coordinates
        loc, _ = self.cv_matcher.find_location(
            self.cv_matcher.findmatch2_png)
        pyautogui.moveTo(loc[0] - 100, loc[1], duration=0.2)
        pyautogui.click(button="left", duration=0.2)

        loc2, thershold = self.cv_matcher.find_location(
            self.cv_matcher.deathmatch2_png)
        if thershold > 0.7:
            pyautogui.moveTo(loc2[0] + 250, loc2[1], duration=0.2)
            pyautogui.click(button="left", duration=0.2)
        else:
            pyautogui.dragTo(loc[0] - 400, loc[1],
                duration=0.2, button="left")
            time.sleep(2)
            loc2, thershold = self.cv_matcher.find_location(
                self.cv_matcher.deathmatch2_png)
        if thershold < 0.6:
            print("no deathmatch game mode")
            exit()
        pyautogui.moveTo(loc2[0] + 250, loc2[1], duration=0.2)
        pyautogui.click(button="left", duration=0.2)

        # find and click the tutorial image
        loc, _ = self.cv_matcher.find_location(
            self.cv_matcher.tutor_png)
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
        _, thres = self.cv_matcher.find_location(
            self.cv_matcher.entertutor2_png)
        if thres > 0.7:
            self.info['waiting'] = False

    def _check_if_game_session_ended(self):
        """Determine if the Tutorial has finished

        Powered by OpenCV template matching
        """
        if not self.info['waiting']:
            _, thershold = self.cv_matcher.find_location(
                self.cv_matcher.finishtutor_png)
            if thershold > 0.7:
                self.done = True

    def _reset_game(self):
        pass
