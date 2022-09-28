import argparse

from data.datasets.multivariate.data_storage import MultivariateEnsembleDataStorage
from data.datasets.univariate.data_storage import VolumeDataStorage

class CVolDataStorageFactory():

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        # TODO implement summarizing parser for Uni/ Multi
        raise NotImplementedError()

    # Factory pattern
    @classmethod
    def from_dict(cls, args):
        prefix = 'data_storage:'
        key = prefix + 'filename_pattern'
        useUnivariate = key in args and args[key].strip() # TODO find better distinguishing property

        if useUnivariate:
            return VolumeDataStorage(
                args[prefix + 'filename_pattern'], base_path=args[prefix + 'base_path'],
                timestep_index=args[prefix + 'timestep:index_range'],
                ensemble_index=args[prefix + 'ensemble:index_range'],
                verify_files_exist=args['verify_files_exist']
            )
        else:
            return MultivariateEnsembleDataStorage(
                timestep_index=args[prefix + 'timestep:index_range'],
                ensemble_index=args[prefix + 'ensemble:index_range'],
                variables=args[prefix + 'variables'], normalization=args[prefix + 'normalization'],
                verify_files_exist=args['verify_files_exist']
            )



def _test_volume_data_storage():
    file_pattern = 'datasets/Ejecta/snapshot_070_256.cvol'#'datasets/ScalarFlow/sim_000000/volume_000100.cvol'
    base_path = ''#'/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications'

    args = {'data_storage:filename_pattern': file_pattern,
            'data_storage:base_path': base_path,
            'data_storage:timestep:index_range': '0:1',
            'data_storage:ensemble:index_range': '0:1',
            'verify_files_exist': True
            }

    vds = CVolDataStorageFactory.from_dict(args)

    volume = vds.load_volume()
    feature = volume.get_feature(0)
    level = feature.get_level(0).to_tensor() # Example of getting Tensor out of pyrenderer.Volume()
    print('Finished')


if __name__ == '__main__':
    _test_volume_data_storage()