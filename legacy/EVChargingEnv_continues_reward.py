import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy


class EVChargingEnv_continues(gym.Env):
    def __init__(self, evs, power_limit):
        super(EVChargingEnv_continues, self).__init__()

        """This function intended to initialize environment variables and load data set

         # Action space is multi-discrete, in this case, there are 3 actions per EV, and actions could be different
        # from each other
        # all the algorithms that supports MultiDiscrete action space support this environment

         # In observation_space 'evs; high limit value is assumed to 300
        # and on power limit / power_allowed high should not reach above 1000 
        """
        self.time = None
        self.total_reward = None

        self.terminated = False
        self.truncated = False

        self.evs_const = evs  # evs dataset
        self.evs = None  # used in reset() to load initial dataset
        self.power_limit = power_limit  # Power limits is surplus power that system can use to charge evs

        self.action_space = spaces.MultiDiscrete([3] * len(evs))
        ev = evs[0]
        num_keys = len(ev.keys())
        NUM_OF_FEATURES = num_keys

        self.observation_space = spaces.Dict({
            'evs': spaces.Box(low=0, high=300, shape=(len(evs), NUM_OF_FEATURES)),
            'power_allowed': spaces.Box(low=0, high=1000, shape=(24,)),
            'time': spaces.Discrete(24)
        })

    def reset(self, **kwargs):
        """ Reset function is intended to reset all environment variables to enable initial
         state for observation and learning of the agent """

        self.terminated = False
        self.truncated = False
        self.time = 0
        self.total_reward = 0  # cumulative reward
        # self.evs = self.evs_const
        self.evs = copy.deepcopy(self.evs_const)
        info = {}

        evs_obs = np.array([[ev['Arrival_time[h]'], ev['TuD (int)'], ev['Battery capacity [KWh]'],
                             ev['ENonD'], ev['SOC']] for ev in self.evs], dtype=np.float32)
        power_allowed_obs = np.array(self.power_limit, dtype=np.float32)
        time_obs = self.time

        observation = {'evs': evs_obs, 'power_allowed': power_allowed_obs, 'time': time_obs}

        episode_info = {"total_reward": self.total_reward, "observation": observation}
        info = {"episode": episode_info}  # Update the info dictionary with the episode information

        return (observation, info)

    def step(self, actions):
        """ Step function intended to execute agent actions:
        in this case the actions that return are 0,1,2, and they converted to charge rate
        this function also responsible to update SOC, TUD, and rewards
        at the end of the step the function return the new state St+1 to agent """

        info = {}  # Additional information for debugging

        charge_actions = [0, 3.7, 11]  # Charge levels that are allowed
        charge = np.zeros(len(actions))
        # fill charge vector in power values based on actions
        for j, action in enumerate(actions):
            charge[j] = charge_actions[action]

        # Immediate rewards - per step
        charge_time_reward = 0
        SOC_reward = 0
        power_reward = 0

        # Update the SOC of each EV in self.evs
        for idx, ev in enumerate(self.evs):

            '''1. Time constraints reward'''

            if (ev['Arrival_time[h]'] < self.time) | (ev['TuD (int)'] == 0):
                if charge_actions[actions[idx]] > 0:

                    #charge[idx] = 0 #
                    charge_time_reward -= 50
                else:
                    charge_time_reward += 1
                # pass

            elif (ev['Arrival_time[h]'] >= self.time) & (ev['TuD (int)'] > 0):
                # Update charge levels based on the selected actions
                ev['SOC'] += charge_actions[actions[idx]] / ev['Battery capacity [KWh]']
                charge_time_reward += 0

            # Update TUD
            if ev['TuD (int)'] > 0:
                ev['TuD (int)'] -= 1

            elif ev['TuD (int)'] <= 0:
                pass

        """ 2. Power constraint"""
        exp_const_pwr = 2
        penalty_const_pwr = 2

        power_reward = 1 - np.exp(
            -exp_const_pwr * (self.power_limit[self.time] - charge.sum()) / self.power_limit[self.time]) #* penalty_const_pwr

        power_reward = penalty_const_pwr * (np.clip(power_reward, a_min=-1000000, a_max=0))

        """3 SOC during episode"""
        ''' for idx, ev in enumerate(self.evs):
            ENonD_SOC = ev['ENonD'] / ev['Battery capacity [KWh]']
            min_SOC = 0.2
            exp_const_soc = 5
            penalty_const_soc = 2

            temp_reward_soc = 1 - np.exp(
                exp_const_soc * (((ENonD_SOC + min_SOC) - ev['SOC']) / (ENonD_SOC + min_SOC)))

            SOC_reward += penalty_const_soc * (np.clip(temp_reward_soc, a_min=-1000000, a_max=0))

        # Update the time and check if the episode is done'''
        self.time += 1

        if self.time == 24:
            self.terminated = True

            for idx, ev in enumerate(self.evs):
                """ 3. SOC level constraint give reward only on departure """
                ENonD_SOC = ev['ENonD'] / ev['Battery capacity [KWh]']

                min_SOC = 0.2
                exp_const_soc = 5
                penalty_const_soc = 2

                temp_reward_soc = 1 - np.exp(exp_const_soc * (((ENonD_SOC + min_SOC) - ev['SOC']) / (ENonD_SOC + min_SOC))) #* penalty_const_soc
               
                #SOC_reward += temp_reward_soc
                SOC_reward += penalty_const_soc * (np.clip(temp_reward_soc, a_min=-1000000, a_max=0))


            evs_obs = np.array(
                [[ev['Arrival_time[h]'], ev['TuD (int)'], ev['Battery capacity [KWh]'], ev['ENonD'], ev['SOC']] for ev
                 in self.evs], dtype=np.float32)
            power_allowed_obs = np.array(self.power_limit, dtype=np.float32)
            time_obs = self.time
            observation = {'evs': evs_obs, 'power_allowed': power_allowed_obs, 'time': time_obs}

        reward = (charge_time_reward + power_reward + SOC_reward)

        # Create the observation for the new time step
        evs_obs = np.array(
            [[ev['Arrival_time[h]'], ev['TuD (int)'], ev['Battery capacity [KWh]'], ev['ENonD'], ev['SOC']] for ev in
             self.evs], dtype=np.float32)

        power_allowed_obs = np.array(self.power_limit, dtype=np.float32)
        time_obs = self.time
        observation = {'evs': evs_obs, 'power_allowed': power_allowed_obs, 'time': time_obs}

        # info output data

        self.total_reward += reward

        info['total_reward'] = self.total_reward
        info['observation'] = observation
        info['charge_pwr'] = np.sum(charge)

        return observation, reward, self.terminated, self.truncated, info


# Define a wrapper function that takes 'evs' and 'power_limit' as arguments and returns the environment instance
def make_ev_charging_env(evs, power_limit):
    return EVChargingEnv_continues(evs=evs, power_limit=power_limit)
