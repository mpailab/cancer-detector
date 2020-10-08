# External imports
import configargparse

# Internal imports
import option
import preselector
import selector

#########################################################################################
# Read script options

parser = configargparse.ArgParser( prog = 'main.py',
                                   description='Some description',
                                   default_config_files=['./.config'])

parser.add( '--config', metavar='path', type=str, is_config_file=True, 
            help='configuration file')
parser.add( '--database', metavar='path', type=str, 
            help='path to database containing gene expression tables')

args = parser.parse_args()

#########################################################################################
# Initialize options

for opt, value in vars(args).items():
    if opt in option.__dict__:
        option.__dict__[opt] = value

#########################################################################################
# Run script

if __name__ == "__main__":
    pass