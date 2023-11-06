import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Space, Discrete, MultiDiscrete, Box
from pymgrid.algos.Control import SampleAverageApproximation
from gym.spaces.space import Space

class Environment(gym.Env):

    def __init__(self, env_config, seed=42):
        #set seed
        np.random.seed(seed)

        self.mg = env_config['building']
        self.Na = 2 + self.mg.architecture['grid'] * 3 + self.mg.architecture['genset'] * 1

        if self.mg.architecture['grid'] == 1 and self.mg.architecture['genset'] == 1:
            self.Na += 1
        self.action_space = Discrete(self.Na)

        # net_load and soc
        self.Ns = 2
        dim1 = int(self.mg.parameters['PC_rated_power'] + self.mg.parameters['load'])
        dim2 = 100
        self.observation_space = MultiDiscrete([dim1, dim2])
        self.metatdata = {"render.modes": ["human"]}
        self.state, self.rewards, self.done, self.info, self.round = None, None, None, None, None
        
        self.seed()
        self.reset()
    
    def get_reward(self):
        return -self.mg.get_cost()
    
    def get_cost(self):
        return sum(self.mg._df_record_cost['cost'])
    
    def step(self, action):
        #update the microgrid
        control_dict = self.get_action(action)
        self.mg.run(control_dict)

        #Compute new state and reward
        self.state = self.transition()
        self.reward = self.get_reward()
        self.done = self.mg.done()
        self.info = {}
        self.round += 1

        return self.state, self.reward, self.done, self.info
    
    def reset(self, testing=False, sampling_args = None):
        self.round = 1
        # Resetting microgrid
        self.mg.reset(testing=testing)

        self.state, self.reward, self.done, self.info = self.transition(), 0, False, {}

        return self.state
    
    def get_action(self, action):
        '''
        :param action: current action
        :return: control_dict : dicco of controls
        
        States are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        '''

        return self.get_action_prioity_list(action)
    
    def states(self):
        observation_space = []
        return observation_space
    
    # Transition function
    def transition(self):
        net_load = round(self.mg.load - self.mg.pv)
        soc = round(self.mg.battery.soc, 1)
        s_ = (net_load, soc) #next state
        return s_
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        txt = "state: {} reward: {} info: {}".format(str(self.state), str(self.reward), str(self.info))
        print(txt)
        
    # Mapping between action and control dict
    def get_action_priority_list(self, action):
        '''
        :param action: current action
        :return: control_dict : dicco of controls
        
        States are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        '''
        mg = self.mg
        pv = mg.pv
        load = mg.load
        net_load = load - pv 
        capa_to_charge = mg.battery_capa_to_charge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(min(net_load, capa_to_charge, p_discharge_max), 0)

        control_dict = {}
        control_dict = self.actions_agent_discret(mg, action)
        return control_dict
    
    def actions_agent_discret(self, mg, action):
        if mg.architecture['genset'] == 1 and mg.architecture['grid'] == 1:
            control_dict = self.action_grid_genset(mg, action)
        else:
            control_dict = self.action_grid(mg, action)

        return control_dict
    
    def action_grid(self, mg, action):
        pv = mg.pv
        load = mg.load
        net_load = load - pv

        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge_pv = max(min(-net_load, capa_to_charge, p_charge_max), 0)
        p_charge_grid = max(min(capa_to_charge, p_charge_max), 0)

        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(min(net_load, capa_to_discharge, p_discharge_max), 0)

        # Charge battery
        if action == 0:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_pv,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': max(pv - min(pv, load) - p_charge_pv, 0),
                            'genset': 0
                            }
        elif action == 4:
            load += p_charge_grid 
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_grid,
                            'battery_discharge': 0,
                            'grid_import': max(load-min(pv, load),0),
                            'grid_export': max(pv - min(pv, load) - p_charge_pv, 0),
                            'genset': 0
                            }
        # Discharge full
        elif action == 1:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': p_discharge,
                            'grid_import': max(0, load - min(pv, load) - p_discharge),
                            'grid_export': 0,
                            'genset': 0
                            }
        #Import load
        elif action == 2:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': max(0, net_load),
                            'grid_export': 0,
                            'genset': 0
                            }
        # Export load
        elif action == 3:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': abs(min(net_load, 0)),
                            'genset': 0
                            }

        return control_dict
    
    def action_grid_genset(self, mg, action):
        pv = mg.pv
        load = mg.load
        net_load = load - pv

        status = mg.grid.status  # whether there is an outage or not
        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge_pv = max(0, min(-net_load, capa_to_charge, p_charge_max))
        p_charge_grid = max(0, min( capa_to_charge, p_charge_max))

        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

        capa_to_genset = mg.genset.rated_power * mg.genset.p_max
        p_genset = max(0, min(net_load, capa_to_genset))

        # Charge
        if action == 0:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_pv,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': max(0, pv - min(pv, load) - p_charge_pv) * status,
                            'genset': 0
                            }
        if action == 5:
            load = load+p_charge_grid

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_grid,
                            'battery_discharge': 0,
                            'grid_import': max(0, load - min(pv, load)) * status,
                            'grid_export': max(0, pv - min(pv, load) - p_charge_grid) * status,
                            'genset': 0
                            }


        # discharge
        elif action == 1:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': p_discharge,
                            'grid_import': max(0, load - min(pv, load) - p_discharge) * status,
                            'grid_export': 0,
                            'genset': 0
                            }

        # Import
        elif action == 2:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': max(0, net_load) * status,
                            'grid_export': 0,
                            'genset': 0
                            }
        # Export
        elif action == 3:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': abs(min(net_load, 0)) * status,
                            'genset': 0
                            }
        # Genset
        elif action == 4:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': 0,
                            'genset': max(net_load, 0)
                            }

        elif action == 6:
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': p_discharge,
                            'grid_import': 0,
                            'grid_export': 0,
                            'genset': max(0, load - min(pv, load) - p_discharge),
                            }
        return control_dict