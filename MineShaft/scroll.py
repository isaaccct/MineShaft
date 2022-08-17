import pyautogui
import time

"""
This is the code of _init__() , which takes multiple parameters, 
but must include a parameter named self as the first parameter

Parameters
----------
value : float
Value parameter of the _mouse_scroll
value parameter is a value from -1 to 1
scroll_step : float
Value parameter times a default of the _mouse_scroll

This is the for loop code of mouse scroll 
in the for loop code,
The range() function defaults scroll_step to be the value
The mouse scroll wheel can be simulated by calling the scroll()function 
and the pyautogui.scroll() function value parameter to be the value
within 0.2 second since calling the function

Returns
-------
The for loop run once time 
it will return once record of _scroll_step
       
"""
class Scroll:

        def __init__(self, value,scroll_step):
                self.value=value
                self.scroll_step=scroll_step


        def _mouse_scroll(self, value,scroll_step):
                for i in range(scroll_step):
                        pyautogui.scroll(value)
                        time.sleep(0.2)


        def _scroll_step(self):
                return self._scroll_step






