import copy
import shutil

from collections import defaultdict
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Union, List

import gymnasium as gym
import numpy as np

from marl_factory_grid.utils.level_parser import LevelParser
from marl_factory_grid.utils.observation_builder import OBSBuilder
from marl_factory_grid.utils.config_parser import FactoryConfigParser
from marl_factory_grid.utils import helpers as h
import marl_factory_grid.environment.constants as c
from marl_factory_grid.utils.results import Result

from marl_factory_grid.utils.states import Gamestate


class Factory(gym.Env):

    @property
    def action_space(self):
        """
        The action space defines the set of all possible actions that an agent can take in the environment.

        :return: Action space
        :rtype: gym.Space
        """
        return self.state[c.AGENT].action_space

    @property
    def named_action_space(self):
        """
        Returns the named action space for agents.

        :return: Named action space
        :rtype: dict[str, dict[str, list[int]]]
        """
        return self.state[c.AGENT].named_action_space

    @property
    def observation_space(self):
        """
        The observation space represents all the information that an agent can receive from the environment at a given
        time step.

        :return: Observation space.
        :rtype: gym.Space
        """
        return self.obs_builder.observation_space(self.state)

    @property
    def named_observation_space(self):
        """
        Returns the named observation space for the environment.

        :return: Named observation space.
        :rtype: (dict, dict)
        """
        return self.obs_builder.named_observation_space(self.state)

    @property
    def params(self) -> dict:
        """
        FIXME LEGACY


        :return:
        """
        import yaml
        config_path = Path(self._config_file)
        config_dict = yaml.safe_load(config_path.open())
        return config_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, config_file: Union[str, PathLike], custom_modules_path: Union[None, PathLike] = None,
                 custom_level_path: Union[None, PathLike] = None):
        """
        Initializes the rl-factory-grid as Gym environment.

        :param config_file: Path to the configuration file.
        :type config_file: Union[str, PathLike]
        :param custom_modules_path: Path to custom modules directory. (Default: None)
        :type custom_modules_path: Union[None, PathLike]
        :param custom_level_path: Path to custom level file. (Default: None)
        :type custom_level_path: Union[None, PathLike]
        """
        self._config_file = config_file
        self.conf = FactoryConfigParser(self._config_file, custom_modules_path)
        # Attribute Assignment
        if custom_level_path is not None:
            self.level_filepath = Path(custom_level_path)
        else:
            self.level_filepath = Path(__file__).parent.parent / h.LEVELS_DIR / f'{self.conf.level_name}.txt'

        parsed_entities = self.conf.load_entities()
        self.map = LevelParser(self.level_filepath, parsed_entities, self.conf.pomdp_r)
        self.levels_that_require_masking = ['two_rooms_small']

        # Init for later usage:
        # noinspection PyTypeChecker
        self.state: Gamestate = None
        # noinspection PyTypeChecker
        self.obs_builder: OBSBuilder = None

        # expensive - don't use; unless required !
        self._renderer = None
        self._recorder = None

        # Init entities
        entities = self.map.do_init()

        # Init rules
        env_rules = self.conf.load_env_rules()
        entity_rules = self.conf.load_entity_spawn_rules(entities)
        env_rules.extend(entity_rules)

        env_tests = self.conf.load_env_tests() if self.conf.tests else []

        # Parse the agent conf
        parsed_agents_conf = self.conf.parse_agents_conf()
        self.state = Gamestate(entities, parsed_agents_conf, env_rules, env_tests, self.map.level_shape,
                               self.conf.env_seed, self.conf.verbose)

        # All is set up, trigger additional init (after agent entity spawn etc)
        self.state.rules.do_all_init(self.state, self.map)

        self.state.tests.do_all_init(self.state, self.map)

        # Build initial observations for all agents
        # noinspection PyAttributeOutsideInit
        self.obs_builder = OBSBuilder(self.map.level_shape, self.state, self.map.pomdp_r)

    def __getitem__(self, item):
        return self.state.entities[item]

    def reset(self) -> (dict, dict):

        # Reset information the state holds
        self.state.reset()

        # Reset Information the GlobalEntity collection holds.
        self.state.entities.reset()

        # All is set up, trigger entity spawn with variable pos
        self.state.rules.do_all_reset(self.state)
        self.state.rules.do_all_post_spawn_reset(self.state)

        # Build initial observations for all agents
        self.obs_builder.reset(self.state)
        return self.obs_builder.build_for_all(self.state)

    def manual_step_init(self) -> List[Result]:
        self.state.curr_step += 1

        # Main Agent Step
        pre_step_result = self.state.rules.tick_pre_step_all(self)
        self.obs_builder.reset(self.state)
        return pre_step_result

    def manual_get_named_agent_obs(self, agent_name: str) -> (List[str], np.ndarray):
        agent = self[c.AGENT][agent_name]
        assert agent, f'"{agent_name}" could not be found. Check the spelling!'
        return self.obs_builder.build_for_agent(agent, self.state)

    def manual_get_agent_obs(self, agent_name: str) -> np.ndarray:
        return self.manual_get_named_agent_obs(agent_name)[1]

    def manual_agent_tick(self, agent_name: str, action: int) -> Result:
        agent = self[c.AGENT][agent_name].clear_temp_state()
        action = agent.actions[action]
        action_result = action.do(agent, self)
        agent.set_state(action_result)
        return action_result

    def manual_finalize_init(self):
        results = list()
        results.extend(self.state.rules.tick_step_all(self))
        results.extend(self.state.rules.tick_post_step_all(self))
        return results

    # Finalize
    def manual_step_finalize(self, tick_result) -> (float, bool, dict):
        # Check Done Conditions
        done_results = self.state.check_done()
        reward, reward_info, done = self.summarize_step_results(tick_result, done_results)

        info = reward_info
        info.update(step_reward=sum(reward), step=self.state.curr_step)
        return reward, done, info

    def step(self, actions):
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        :param actions: An action or list of actions provided by the agent(s) to update the environment state.
        :return: observation, reward, terminated, truncated, info, done
        :rtype: tuple(list(np.ndarray), float, bool, bool, dict, bool)
        """

        if not isinstance(actions, list):
            actions = [int(actions)]

        # --> Action

        # Apply rules, do actions, tick the state, etc...
        tick_result = self.state.tick(actions)

        # Check Done Conditions
        done_results = self.state.check_done()
        done_tests = self.state.tests.check_done_all(self.state)

        # Finalize
        reward, reward_info, done = self.summarize_step_results(tick_result, done_results)

        info = dict(reward_info)

        info.update(step_reward=sum(reward), step=self.state.curr_step)

        obs = self.obs_builder.build_for_all(self.state)
        return None, [x for x in obs.values()], reward, done, info

    def summarize_step_results(self, tick_results: list, done_check_results: list) -> (int, dict, bool):
        # Returns: Reward, Info
        rewards = defaultdict(lambda: 0.0)

        # Gather per agent environment rewards and
        # Combine Info dicts into a global one
        combined_info_dict = defaultdict(lambda: 0.0)
        for result in chain(tick_results, done_check_results):
            assert result, 'Something returned None...'
            if result.reward is not None:
                try:
                    rewards[result.entity.name] += result.reward
                except AttributeError:
                    rewards['global'] += result.reward
            infos = result.get_infos()
            for info in infos:
                assert isinstance(info.value, (float, int))
                combined_info_dict[info.identifier] += info.value

        # Check Done Rule Results
        try:
            done_reason = next(x for x in done_check_results if x.validity)
            done = True
            self.state.print(f'Env done, Reason: {done_reason.identifier}.')
        except StopIteration:
            done = False

        if self.conf.individual_rewards:
            global_rewards = rewards['global']
            del rewards['global']
            reward = [rewards[agent.name] for agent in self.state[c.AGENT]]
            reward = [x + global_rewards for x in reward]
            self.state.print(f"Individual rewards are {dict(rewards)}")
            return reward, combined_info_dict, done
        else:
            reward = sum(rewards.values())
            self.state.print(f"reward is {reward}")
        return reward, combined_info_dict, done

    # noinspection PyGlobalUndefined
    def render(self, mode='human'):
        if not self._renderer:  # lazy init
            from marl_factory_grid.utils.renderer import Renderer
            global Renderer
            self._renderer = Renderer(self.map.level_shape, view_radius=self.conf.pomdp_r, fps=10)

        # Remove potential Nones from entities
        render_entities_full = self.state.entities.render()

        # Hide entities where certain conditions are met (e.g., amount <= 0 for DirtPiles)
        maintain_indices = self.filter_entities(self.state.entities)
        if maintain_indices:
            render_entities = [render_entity for idx, render_entity in enumerate(render_entities_full) if idx in maintain_indices]
        else:
            render_entities = render_entities_full

        # Mask entities based on dynamic conditions instead of hardcoding level-specific logic
        if self.conf['General']['level_name'] in self.levels_that_require_masking:
            render_entities = self.mask_entities(render_entities)

        if self.conf.pomdp_r:
            for render_entity in render_entities:
                if render_entity.name == c.AGENT:
                    render_entity.aux = self.obs_builder.curr_lightmaps[render_entity.real_name]
        return self._renderer.render(render_entities, self._recorder)

    def filter_entities(self, entities):
        """ Generalized method to filter out entities that shouldn't be rendered. """
        if 'CoinPiles' in self.state.entities.keys():
            all_entities = [item for sublist in [[e for e in entity] for entity in entities] for item in sublist]
            return [idx for idx, entity in enumerate(all_entities) if not ('CoinPile' in entity.name and entity.amount <= 0)]

    def mask_entities(self, entities):
        """ Generalized method to mask entities based on dynamic conditions. """
        for entity in entities:
            if entity.name == 'CoinPiles':
                entity.name = 'Destinations'
                entity.value = 1
                #entity.mask = 'Destinations'
                #entity.mask_value = 1
        return entities

    def set_recorder(self, recorder):
        self._recorder = recorder

    def summarize_header(self):
        header = {'rec_step': self.state.curr_step}
        for entity_group in (x for x in self.state if x.name in ['Walls', 'DropOffLocations', 'ChargePods']):
            header.update({f'rec{entity_group.name}': entity_group.summarize_states()})
        return header

    def summarize_state(self):
        summary = {'step': self.state.curr_step}

        # Todo: Protobuff Compatibility Section                                      #######
        #  for entity_group in (x for x in self.state if x.name not in [c.WALLS, c.FLOORS]):
        for entity_group in self.state:
            summary.update({entity_group.name.lower(): entity_group.summarize_states()})
        # TODO Section End                                                          ########
        for key in list(summary.keys()):
            if key not in ['step', 'walls', 'doors', 'agents', 'items', 'dirtPiles', 'batteries', 'coinPiles']:
                del summary[key]
        return summary

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._config_file, filepath)
