import argparse


class IDataStorage(object):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, args):
        raise NotImplementedError()

    def load_volume(self, timestep=None, ensemble=None, variable=None, index_access=False):
        raise NotImplementedError()

    def num_timesteps(self):
        raise NotImplementedError()

    def num_members(self):
        raise NotImplementedError()

    def num_ensembles(self):
        raise NotImplementedError()

    def num_variables(self):
        raise NotImplementedError()


