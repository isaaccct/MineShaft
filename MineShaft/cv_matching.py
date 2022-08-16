import cv2 as cv
import pyautogui
import numpy as np


class cv_matching:
    def __init__(self):
        pass

    def preload_templates(self):
        """
        change the current directory to the script folder so that relative
        path can be used
        """
        folder_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(folder_path)

        # opencv read the find match image by file path
        self.findmatch2_png = cv2.imread(
            "../SourcePictures/findmatch2.png",
            cv2.IMREAD_UNCHANGED)
        self.deathmatch2_png = cv2.imread(
            "../SourcePictures/deathmatch2.png",
            cv2.IMREAD_UNCHANGED)
        self.tutor_png = cv2.imread(
            "../SourcePictures/tutor.png",
            cv2.IMREAD_UNCHANGED)
        self.entertutor2_png = cv2.imread(
            "../SourcePictures/entertutor2.png",
            cv2.IMREAD_UNCHANGED)
        self.finishtutor_png = cv2.imread(
            "../SourcePictures/finishtutor.png",
            cv2.IMREAD_UNCHANGED)

    def find_location(self, tofind):
        """
        the method requires an image parameter which is in numpy.ndarray format
        then it compares the image with screen capture
        """

 
        '''
        screen capture using pyautogui 
        '''
        screen = pyautogui.screenshot()

        '''
        change the capture to numpy array
        '''
        img = np.array(screen)

        '''
        since the matchtemplate() function requires 2 images with same format
        so the below code is to convert the images to RBG format
        '''

        tofind = cv.cvtColor(tofind, cv.COLOR_BGR2RGB)



        background = cv.cvtColor(img, cv.COLOR_BGR2RGB)


        '''
        the below opencv function matchTemplate() receive 3 parameters, first 2 are images in 
        numpy array, the third one is the opencv comparison algorithm. The function returns
        a grayscale image, where each pixel denotes how much does the neighbourhood of that pixel 
        match with template.
        '''
        result = cv.matchTemplate(background,tofind,cv.TM_CCOEFF_NORMED)
        '''
        the below minMaxLoc() function receives a parameter which is a grayscale image
        then return 4 parameters:
        1. the  thershold value of minimum match objects
        2. the  thershold value of maximum match objects
        3. the  coordinates of minimu match objects
        4. the  coordinates of maximum match objects
        '''
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
        '''
        the methond only returns the thershold value and location of the most suitable matching object
        '''
        return max_loc,max_val








