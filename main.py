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
parser.add( '--datasets', metavar='list', type=str, 
            help='set a list of datasets. Put \'list\' when you need to print all avaliable datasets.')

def par_list (par):
    return [] if par == 'list' else par.split()

parser.add( '--preselectors', metavar='list', type=par_list, default=preselector.names(),
            help='set a list of preselectors. Put \'list\' when you need to print all avaliable preselectors.')
parser.add( '--selectors', metavar='list', type=par_list, default=selector.names(), 
            help='set a list of selectors. Put \'list\' when you need to print all avaliable selectors.')

args = parser.parse_args()

#########################################################################################
# Initialize options

is_exit = False

for opt, value in vars(args).items():

    if opt == 'preselectors':

        if not value:
            print('All avaliable preselectors:\n  ' + '\n  '.join(preselector.names()))
            is_exit = True

        else:
            for x in value:
                if x not in preselector.names():
                    print('warning: unknown preselector %r will be skipped.' % x)

            value = [ preselector.get(x) for x in value if x in preselector.names() ]
            if not value:
                print('warning: the empty list of preselectors, use all valiable preselectors.')
                value = preselector.funcs()
            
    elif opt == 'selectors':

        if not value:
            print('All avaliable selectors:\n  ' + '\n  '.join(selector.names()))
            is_exit = True

        else:
            for x in value:
                if x not in selector.names():
                    print('warning: unknown selector %r will be skipped' % x)

            value = [ selector.get(x) for x in value if x in selector.names() ]
            if not value:
                print('warning: the empty list of selectors, use all valiable selectors.')
                value = selector.funcs()

    if opt in option.__dict__:
        option.__dict__[opt] = value

if is_exit:
    exit(0)

#########################################################################################
# Run script

if __name__ == "__main__":
    print(option.preselectors)
    print(option.selectors)
    pass