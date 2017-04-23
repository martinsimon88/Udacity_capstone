import os
import sys
import numpy as np

from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace

def main():

    options         = parse_config_file('.', 'config.json')
    experiment_name = options["experiment-name"]
    input_space     = InputSpace(options["variables"])
    db              = MongoDB(database_address=options['database']['address'])

    i = 0
    recommendation = db.load(experiment_name, 'recommendations', {'id' : i + 1})
    while recommendation is not None:
        params_last = input_space.vectorify(recommendation[ 'params' ])
        recommendation = db.load(experiment_name, 'recommendations', {'id' : i + 1})
        i += 1

    np.savetxt('pareto_front.txt', params_last, fmt = '%e')

if __name__ == '__main__':
    main()
