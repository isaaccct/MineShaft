"""This is the base class of the MineShaft environment.
It inherited OpenAI Gym Environment.

For any game supported by MineShaft,
the environment inherited this class as an interface.
"""
from enum import Enum

import gym
from gym import spaces

class BaseEnv(gym.Env):
    """Custom Environment that follows gym interface
    
    By default, the environment let reinforcement learning agent play
    the game in the way of human (screen, keyboard and mouse) and giving
    it the full control of the game from setting to character selection
    and weapon selection and joining the match.
    
    It is assumed that the reinforcement learning agent know what its
    going to do and do pick the right tools for its plan.
    
    > :warning: The major difference between MineShaft environment and
    >           OpenAI Gym is that even the `step()` may not be called,
    >           the game in this environment will continue. i.e. The
    >           time interval between each `step()` may not be the same.
    >           The game does not pause like what OpenAI Gym do.
    
    Attributes
    ----------
    IO_MODE : Enum
        IO_MODE.API = direct access to game API
        IO_MODE.SIMPLIFIED = simpilified observation_space and action_space
        IO_MODE.FULL_CONTROL = screen and full keyboard and mouse control
    EXPLORE_MODE : Enum
        EXPLORE_MODE.MATCH = reinforcement learning agent
                             only take control during match
        EXPLORE_MODE.FULL = reinforcement learning agent
                             have full control since login
    """
    metadata = {'render.modes': ['human']}
    IO_MODE = Enum('IO_MODE', ('API', 'SIMPLIFIED', 'FULL_CONTROL'))
    EXPLORE_MODE = Enum('EXPLORE_MODE', ('MATCH', 'FULL'))

    def __init__(self, io_mode=IO_MODE.FULL_CONTROL,
                 explore_space=EXPLORE_MODE.FULL):
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
        super(BaseEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        if io_mode == IO_MODE.FULL_CONTROL:
            # press & release channels; 92 key + mouse (move + click + scroll)
            ACTION_SHAPE = (2, 92 + (2 + 2 + 1))
            # screen height
            HEIGHT = 512
            # screen width
            WIDTH = 512
            # screen color channels
            N_CHANNELS = 3
        elif io_mode == IO_MODE.SIMPLIFIED:
            # keyboard (WASD/SPACE/12345) + mouse (move + click + scroll)
            ACTION_SHAPE = (2, 11 + (2 + 2 + 1))
            # screen height
            HEIGHT = 512
            # screen width
            WIDTH = 512
            # screen color channels
            N_CHANNELS = 3
        elif io_mode == IO_MODE.API:
            raise NotImplementedError

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=ACTION_SHAPE, dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(HEIGHT, WIDTH, N_CHANNELS),
                                            dtype=np.uint8)
        self.explore_space = explore_space
        return self.action_space, self.observation_space

    def step(self, action):
        """Send keyboard and mouse action to the game environment
        
        This function should be triggered every 0.2s in ideal condition.
        In case it was not triggered, the game will continue going without
        pausing and waiting for new command like OpenAI Gym do. Therefore
        it is sensitive to the time consumption in the implementation of
        reinforcement learning agent.
        
        Parameters
        ----------
        action :
            Composed of two matrices indicating the action is a key down
            or key up event of keyboard and mouse. The composed matrix has
            the same shape as `self.action_space`. For any value greater
            than 0 in the first matrix, environment take it as key down
            event. For any value greater than 0 in the second matrix the
            environment take it as key up event. The only exception is
            the matrix elements of mouse move and mouse scroll, the value
            means `(x, y)` on screen (move to `int(x * screen_width),
            int(y * screen_height)`) and `z` percentage of scroll wheel
            on up and down.

        Returns
        -------
        observation : numpy.array
            Screen capture from the environment or data captured from game's
            Application Programming Interface.
        reward : float32
            Score, how good is the step performed by reinforcement learning
            agent in gaining advantage to win the game in the step. Or scoring
            the reinforcement learning agent with all the steps performed that
            bring it to the current situation.
        done : boolean
            Is the game finished and require a reset of the environment to start
            another game or `False` that `step()` should be called and continue
            the game? `True` or `False`.
        info : dict
            Extra information in `dict` data structure to be used by human,
            reinforcement learning agent developer or the reinforcement learning
            agent. It can be an empty dictionary if not used.
        """
        self._take_action(action)
        reward = self._get_reward()
        ob = np.array(sct.grab(monitor))
        episode_over = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        """Reset the environment after `done == False` returned from `step()` to
        start a new game.
        
        Returns
        -------
        observation : numpy.array
            Screen capture from the environment or data captured from game's
            Application Programming Interface. May serve as the first observation
            from environment for the reinforcement learning agent to generate its
            first output as `action` to `step()`
        """
        observation = self.observation_space
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        """Show the game window.
        
        Actually we do show the window by default and have not found a way to
        hide it from the screen, so this function is not applicable to MineShaft
        environment at the moment.
        
        Parameters
        ----------
        mode : str
            I have no idea of this
            
        .. deprecated:: 0.0.1
        """
        pass

    def close (self):
        """Close the game and release the system memory.
        """
        pass
