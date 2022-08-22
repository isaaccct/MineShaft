import win32api
import time
import unittest

import numpy as np
import pyautogui
import keyboard

from MineShaft.ThetanArenaEnv import ThetanArenaEnv

class ThetanArenaEnvTestCase(unittest.TestCase):
	def setUp(self):
		self.mouse_x_y = (0.3, 0.6)
		self.env = ThetanArenaEnv()

	def tearDown(self):
		self.mouse_x_y = None

	def test__mouse_move(self):
		"""Test case for mouse move method
		
		The function will call mouse move method of `ThetanArenaEnv` with
		parameters `x=0.3, y=0.6` of the screen. Then check if the mouse
		coordinate now moved to the expected `(x, y)` location.
		
		(Cannot test correctly if `(x, y)` definition changed to reletive
		coordinate of game window)
		"""
		expected = np.asarray(pyautogui.size()) * np.asarray(self.mouse_x_y);
		self.env._mouse_move(self.mouse_x_y);
		result = np.asarray(pyautogui.position());
		self.assertTrue(np.abs(expected - result) < 2);

	def test__mouse_click(self):
		"""Test case for mouse clikc method

		Use `win32api` to determine if the press & release works.
		"""
		self.env._mouse_click(np.asarray([0, 0.4]));
		self.env._mouse_click(np.asarray([0.5, -0.4]));
		self.env._mouse_click(np.asarray([0.6, 0.6]));
		self.env._mouse_click(np.asarray([0, -0.4]));

		state_left = win32api.GetKeyState(0x01);
		self.env._mouse_press(True, False);
		state_a = win32api.GetKeyState(0x01);
		self.assertTrue(state_left != state_a);
		self.assertTrue(state_a < 0);
		self.env._mouse_release(True, False);
		state_a = win32api.GetKeyState(0x01);
		self.assertTrue(state_left != state_a);
		self.assertTrue(state_a >= 0);

		state_right = win32api.GetKeyState(0x02);
		self.env._mouse_press(False, True);
		state_a = win32api.GetKeyState(0x02);
		self.assertTrue(state_right != state_a);
		self.assertTrue(state_a < 0);
		self.env._mouse_release(False, True);
		state_a = win32api.GetKeyState(0x02);
		self.assertTrue(state_right != state_a);
		self.assertTrue(state_a >= 0);

	def test__mouse_drag(self):
		"""Test case for mouse drag (combination of mouse action)

		Use `pyautogui.position` to determine if the move works.
		Use `win32api` to determine if the press & release works.
		"""
		self.env._mouse_move((0, 0));
		result = np.asarray(pyautogui.position());
		self.assertTrue(np.abs(np.zeros(2) - result) < 2);
		state_left = win32api.GetKeyState(0x01);
		self.env._mouse_press(True, False);
		state_a = win32api.GetKeyState(0x01);
		self.assertTrue(state_left != state_a);
		self.assertTrue(state_a < 0);
		expected = np.asarray(pyautogui.size()) * np.asarray(self.mouse_x_y);
		self.env._mouse_move(self.mouse_x_y);
		result = np.asarray(pyautogui.position());
		self.assertTrue(np.abs(expected - result) < 2);
		self.env._mouse_release(True, False);
		state_a = win32api.GetKeyState(0x01);
		self.assertTrue(state_left != state_a);
		self.assertTrue(state_a >= 0);

		state_right = win32api.GetKeyState(0x02);
		self.env._mouse_press(False, True);
		state_a = win32api.GetKeyState(0x02);
		self.assertTrue(state_right != state_a);
		self.assertTrue(state_a < 0);
		expected = np.asarray(pyautogui.size()) * np.asarray(self.mouse_x_y);
		self.env._mouse_move(self.mouse_x_y);
		result = np.asarray(pyautogui.position());
		self.assertTrue(np.abs(expected - result) < 2);
		self.env._mouse_release(False, True);
		state_a = win32api.GetKeyState(0x02);
		self.assertTrue(state_right != state_a);
		self.assertTrue(state_a > 0);
		
	def test__keyboard_input(self):
		"""Test case for keyboard input

		The test case will try to press "a" and "ctrl + a".
		And use `keyboard` to detect if the key is pressed as the function
		of `_keyboard_press` is designed to be.

		Notes
		-------
			The keyboard input action is a 1D matrix of float value with
			length `80`. Each value in the matrix represent the force
			pressed on the following keys on the keyboard (in the same
			sequence).
			```
			[
				'altleft', 'altright', 'ctrlleft', 'ctrlright', 'shiftleft',
				'shiftright', 'backspace', 'capslock', 'delete', 'down',
				"'", ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
				'6', '7', '8', '9', ';', '=', '[', '\\', ']', '`', 'a',
				'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j', 'k', 'l',
				'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
				'x', 'y', 'z', 'insert', 'left', 'num0', 'num1', 'num2',
				'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9',
				'end', 'enter', 'esc', 'numlock', 'pagedown', 'pageup',
				'right', 'space', 'tab', 'up', 'home'
			]
			```
			The value of force will be in range `-1` to `1` and it is considered
			to be pressed down only if the value is larger than `0`.
		"""
		action_a = np.zeros((80));
		action_a[31] = 1;
		# press "a"
		self.env._keyboard_press(action_a);
		assert keyboard.is_pressed("a");
		self.env._keyboard_release(action_a);
		assert not keyboard.is_pressed("a");
		action_ctrl_a = np.zeros((80));
		action_ctrl_a[31] = 1;
		action_ctrl_a[2] = 1;
		# press Ctrl+a
		self.env._keyboard_press(action_ctrl_a);
		assert keyboard.is_pressed('ctrl+a');
		self.env._keyboard_release(action_ctrl_a);
		assert not keyboard.is_pressed('ctrl+a');
