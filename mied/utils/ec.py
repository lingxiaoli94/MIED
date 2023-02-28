import torch
import argparse
import yaml
import copy
from pathlib import Path
import shutil
from datetime import datetime
from uuid import uuid4
from collections import namedtuple
import wandb

from mied.utils.shortname import \
    convert_method_cls_to_str, convert_method_str_to_cls, \
    convert_projector_cls_to_str, convert_projector_str_to_cls


class Config:
    def __init__(self, param_dict):
        self.param_dict = copy.copy(param_dict)


    def has_same_params(self, other):
        return self.param_dict == other.param_dict


    def __getitem__(self, k):
        return self.param_dict[k]


    def get(self, k, default_v):
        return self.param_dict.get(k, default_v)


    def __repr__(self):
        return str(self.param_dict)


    @staticmethod
    def from_yaml(yaml_path):
        return Config(yaml.safe_load(open(yaml_path, 'r')))


    def save_yaml(self, yaml_path):
        open(yaml_path, 'w').write(yaml.dump(self.param_dict))



class ConfigBlueprint:
    def __init__(self, default_param_dict):
        '''
        :param default_param_dict: a dict, where the values have to be one of
        [string, int, float]
        '''
        self.default_param_dict = default_param_dict


    def prepare_parser(self, parser):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        for k, v in self.default_param_dict.items():
            if type(v) == bool:
                parser.add_argument('--{}'.format(k), type=str2bool, default=v)
            else:
                parser.add_argument('--{}'.format(k), type=type(v), default=v)


ECParseResult = namedtuple('ECParseResult',
                           ['tmp_args', 'config', 'exp_dir'])


class ExperimentCoordinator:
    def __init__(self, root_dir):
        '''
        We assume the following hierarchy of directories:
        root_dir/exps/exp_name/...
            - conf.yml: the configuration corresponding to an instance of
            Config class
            Then arbitrary files and subfolders can be placed here, e.g.,
                - result.h5: result in hdf5 format
                - log/: tensorboard log
                - ckpt.tar: checkpoint
        When possible, we assume each conf.yml corresponds to a
        unique exp_name.

        :param root_dir: root directory of the experiments
        '''
        self.root_path = Path(root_dir)

        # Temporary blueprints are non-persistent.
        self.temporary_blueprints = [ConfigBlueprint({
            'device': 'cuda',
            'val_freq': 100,
        })]

        self.common_blueprints = [ConfigBlueprint({
            'project': 'uncategorized',
            'wandb': True,
            'seed': 42,
            'optimizer': 'Adam',
            'lr': 1e-2,
            'beta1': 0.9,
            'beta2': 0.999,
            # Every method is particle-based.
            'num_particle': 50,
            'precondition': False,
        })]
        self.method_blueprint_dict = {}
        self.projector_blueprint_dict = {}


    def add_temporary_arguments(self, param_dict):
        self.temporary_blueprints.append(ConfigBlueprint(param_dict))


    def add_common_arguments(self, param_dict):
        self.common_blueprints.append(ConfigBlueprint(param_dict))


    def add_method_arguments(self, method_cls, param_dict):
        self.method_blueprint_dict[method_cls] = ConfigBlueprint(param_dict)


    def add_projector_arguments(self, projector_cls, param_dict):
        self.projector_blueprint_dict[projector_cls] = ConfigBlueprint(param_dict)


    def parse_args(self):
        tmp_parser = argparse.ArgumentParser()
        '''
            * --resume: continue an experiment (the corresponding folder
            must have a conf.yml file)
            * --exp_name: name of the experiment which is the same as the
            folder name containing this experiment's related files. If not
            provided, a random unique name will be generated (which can later
            be changed).
        '''
        tmp_parser.add_argument('--resume', type=str)
        tmp_parser.add_argument('--override', action='store_true', default=False)
        tmp_parser.add_argument('--restart', action='store_true', default=False)
        tmp_parser.add_argument('--exp_name', type=str)
        for b in self.temporary_blueprints:
            b.prepare_parser(tmp_parser)
        tmp_args, _ = tmp_parser.parse_known_args()
        if tmp_args.resume:
            assert(tmp_args.exp_name is None)
            exp_dir = self.get_exps_path() / Path(tmp_args.resume)
            config = Config.from_yaml(exp_dir / 'conf.yml')
            print('Resuming experiment {}...'.format(exp_dir))
        else:
            common_parser = argparse.ArgumentParser()
            common_parser.add_argument('--method', type=str, default='RED')
            common_parser.add_argument('--projector', type=str, default='DB')

            for b in self.common_blueprints:
                b.prepare_parser(common_parser)
            common_args, _ = common_parser.parse_known_args()

            method_cls = convert_method_str_to_cls(common_args.method)
            projector_cls = convert_projector_str_to_cls(common_args.projector)
            if method_cls not in self.method_blueprint_dict:
                raise Exception('Cannot find blueprint for '
                                f'method {method_cls}!')
            if projector_cls not in self.projector_blueprint_dict:
                raise Exception('Cannot find blueprint for '
                                f'projector {method_cls}!')

            method_parser = argparse.ArgumentParser()
            self.method_blueprint_dict[method_cls].prepare_parser(
                method_parser
            )
            method_args, _ = method_parser.parse_known_args()
            projector_parser = argparse.ArgumentParser()
            self.projector_blueprint_dict[projector_cls].prepare_parser(
                projector_parser
            )
            projector_args, _ = projector_parser.parse_known_args()

            config_dict = vars(common_args)
            config_dict['method_config'] = vars(method_args)
            config_dict['projector_config'] = vars(projector_args)
            config_dict['wandb_id'] = wandb.util.generate_id()

            config = Config(config_dict)
            exp_dir = self.make_persistent(config, tmp_args.exp_name,
                                           override=tmp_args.override)

        self.parse_result = ECParseResult(
            tmp_args=tmp_args,
            config=config,
            exp_dir=exp_dir
        )
        return self.parse_result


    def create_solver(self, problem):
        config = self.parse_result.config
        exp_dir = self.parse_result.exp_dir
        tmp_args = self.parse_result.tmp_args

        wandb.init(
            project=config['project'],
            mode='online' if config['wandb'] else 'offline',
            config={
                'exp_dir': exp_dir,
                **config.param_dict
            },
            name=('' if tmp_args.exp_name is None else f'{tmp_args.exp_name}'),
            id=config['wandb_id'],
            resume='allow'
        )

        projector_cls = convert_projector_str_to_cls(config['projector'])
        projector = projector_cls(**config['projector_config'])

        method_cls = convert_method_str_to_cls(config['method'])

        solver = method_cls(problem=problem,
                            projector=projector,
                            num_particle=config['num_particle'],
                            precondition=config['precondition'],
                            val_freq=self.parse_result.tmp_args.val_freq,
                            ckpt_path=exp_dir / 'ckpt.tar',
                            logger_fn=lambda d: wandb.log(d),
                            optimizer_conf={
                                'cls': config['optimizer'],
                                'lr': config['lr'],
                                'beta1': config['beta1'],
                                'beta2': config['beta2']
                            },
                            **config['method_config'])
        if not self.parse_result.tmp_args.restart:
            solver.load_ckpt()
        return solver


    def get_exps_path(self):
        path = self.root_path / 'exps/'
        path.mkdir(exist_ok=True)
        return path


    def make_persistent(self, config, exp_name, override):
        exist = False

        # Check if params match any existing conf.yml.
        for p in self.get_exps_path().iterdir():
            if p.is_dir():
                another_exp_name = p.stem
                config_path = p / 'conf.yml'
                if not config_path.exists():
                    continue
                if exp_name == another_exp_name:
                    another_config = Config.from_yaml(config_path)
                    print(f'Found existing experiment {exp_name}!')
                    diff = False
                    for k, v in config.param_dict.items():
                        if k not in another_config.param_dict:
                            print(f'Existing config missing {k}!')
                            diff = True
                        elif another_config[k] != v:
                            print(f'Existing config has {k}={another_config[k]}'
                                  f' whereas new config has {k}={v}!')
                            diff = True
                    for k in another_config.param_dict:
                        if k not in config.param_dict:
                            print(f'New config missing {k}!')
                            diff = True

                    if not override:
                        override = input("Override? [Y/N]")
                    if override == True or override == 'Y':
                        shutil.rmtree(p)
                        exist = False
                        break
                    if diff:
                        raise Exception('Found config with same name'
                                        ' but different parameters! Abort.')
                    print('Resuming experiment {} with '.format(p) +
                          'identical config...')
                    exist = True
                    config = another_config
                    exp_dir = p

        if not exist:
            # Save config
            if exp_name is None:
                exp_name = config['wandb_id']
            exp_dir = self.get_exps_path() / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            config.save_yaml(exp_dir / 'conf.yml')
            print('Saved a new config to {}.'.format(exp_dir))

        return exp_dir

