import argparse

from data.datasets.multivariate.data_storage import MultivariateEnsembleDataStorage, VARIABLE_NAMES
from data.datasets.univariate.data_storage import VolumeDataStorage

class CVolDataStorageFactory():

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('VolumeDataStorage')
        example_pattern = ('"ScalarFlow/sim_{'
                           + VolumeDataStorage.FILE_PATTERN_ENSEMBLE_KEY
                           + ':06d}/density_{'
                           + VolumeDataStorage.FILE_PATTERN_TIME_KEY
                           + ':06d}.cvol"')
        prefix = '--data-storage:'
        group.add_argument(prefix + 'filename-pattern', type=str, help=f"""
                file name pattern used to generate volume file names. 
                To specify time and ensemble indices, the following keys are used:

                time: {VolumeDataStorage.FILE_PATTERN_TIME_KEY}
                ensemble: {VolumeDataStorage.FILE_PATTERN_ENSEMBLE_KEY}

                Example: {example_pattern}
                """) #M: dont't set required = True, since this is now distinguishing property to Multivariate
        group.add_argument(prefix + 'base-path', type=str, default=None, help="""
                base path to be prepended in front of file name pattern
                """)
        group.add_argument(prefix + 'ensemble:index-range', type=str, default=None, help="""
                    Ranges used for the ensemble index. The indices are obtained via
                    <code>range(*map(int, {ensemble_index_range}.split(':')))</code>

                    Example: "0:10:2" (Default: "0:1")
                """)
        group.add_argument(prefix + 'timestep:index-range', type=str, default=None, help="""
                    Ranges used for the keyframes for time interpolation. 
                    At those timesteps, representative vectors are generated, optionally trained,
                    and interpolated between timesteps
                    The indices are obtained via <code>range(*map(int, {timestep-index-range}.split(':')))</code>

                    Example: "0:10:2" (Default: "0:1")
                """)
        group.add_argument(prefix + 'variables', type=str, default=None, help=f"""
                    Variable keys separated by ':'
                    Choices: {VARIABLE_NAMES}
                """)
        group.add_argument(prefix + 'normalization', type=str, default=None, help=f"""
                    normalization type to use
                    Choices: {VARIABLE_NAMES}
                """)
        group.add_argument(prefix + 'disable-file-verification', action='store_false', dest='verify_files_exist', help="""
                disable file verification for performance reasons
                """)
        group.set_defaults(verify_files_exist=True)

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