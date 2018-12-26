import time
import numpy as np
import quad_vrep_env as quad_env

def main():
    env = quad_env.QuadVrepEnv(1000)
    env.reset()

    while True:
        try:
            state,rew,done,extra = env.step(np.zeros(4))
            if done:
                env.reset()
        except(KeyboardInterrupt,SystemExit):
            raise

if __name__ == '__main__':
    main()
