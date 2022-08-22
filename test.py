from stable_baselines3 import PPO

from MineShaft import ThetanArenaEnv

def main():
	env = ThetanArenaEnv()
	env.enter_match(random_character=True)
	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=10_000)

	info = {'waiting': True}
	action = env.action_space.sample()
	for _ in range(1000):
		if not info['waiting']:
			action, _states = model.predict(observation, deterministic=True)
		observation, reward, done, info = env.step(action)
		if done:
			observation = env.reset()

	env.close()


if __name__ == '__main__':
	main()
