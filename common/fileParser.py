import argparse


# M: Use with 'parser.add_argument('--config', type=open, action=LoadFromFile)'
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)