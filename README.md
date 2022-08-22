# MineShaft
An Gym compatible environment for Artificial Intelligence Reinforcement Agent to play GameFi

> ⚠️ Currently support Windows 10 only

## Getting started (user)
```py3
from MineShaft import MineShaft

env = MineShaft()
env.enter_match(random_character=True)
for _ in range(1000):
  observation, reward, done, info = env.step(env.action_space.sample())
env.close()
```

## Getting started (developer)
Install dependencies
```bash
python3 -m pip install -r requirenments.txt
```

### Test Env with `PPO` (developer)
```bash
python3 test.py
```

Open your favourite editor and change anything to see what will happen.

> ⚠️ Before submitting pull request, please read instructions in [CONTRIBUTING.md](CONTRIBUTING.md)
to prevent reject of pull request ( 🚧 `400 Bad Request`) and make both of us happy ☕.

## Functional requirements
### Initialize environment
- [x] Find and start Thetan Arena in Windows. Throw an error if Thetan Arena cannot be found installed in the system.
- [x] Find and capture part of the screen only the game window of Thetan Arena.
### Input action
- [ ] Input given key as keyboard press or keydown or keyup event.
- [x] Move the mouse to the given coordinate.
- [x] Input right click or left click event.
### Start game
- [x] With the help of computer vision and “autogui” keyboard/mouse control (Robotic Process Automation), enter a match with a specified character.
- [ ] Determine if the game has started.
- [ ] Determine if the game has finished.
### Generate random action
- [x] Random movement with keyboard input if the game has started.
- [x] Random fire with mouse input if the game has started.
### Get game state
- [x] Capture part of the screen only the game window of Thetan Arena.
- [ ] The frame-rate of screen capture must be more than 30 frames-per-second.
### Terminate environment
- [x] Quit on-going game and exit the program Thetan Arena. And then destroy the MineShaft class instance itself.

## Supported GameFis
- Thetan Arena
