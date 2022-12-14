import argparse
import pandas as pd
from time import time
from pm4py.objects.log.importer.xes import importer as xes_importer
from importing import import_attributes, convert_attributes, import_petri_net
from model_utils import get_activities_to_transitions_map
from algo import MultiplePaths
from training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str)    # parse arguments
    args = parser.parse_args()
    net_name = args.net_name

    # Importing the log
    log = xes_importer.apply('logs/log-{}.xes'.format(net_name))

    # Importing the attributes_map file
    attributes_map = import_attributes(net_name)

    # Converting attributes types according to the attributes_map file
    log = convert_attributes(attributes_map, log)

    # import model
    petri_net, init_mark, final_mark = import_petri_net(net_name)

#    gviz = pn_visualizer.apply(net, im, fm)
#    pn_visualizer.view(gviz)

    # Dealing with loops and other stuff... needs cleaning
    activities_to_trans_map = get_activities_to_transitions_map(petri_net)
    # TODO declare algo type

    tic = time()
    # Scanning the log to get the logs related to decision points
    print('Extracting training logs from Event Log...')
    algorithm = MultiplePaths(petri_net, activities_to_trans_map)
    training_data, complexity = algorithm.old_extract_decision_points_data(log)
    complexity_raw = pd.DataFrame.from_dict(complexity)
    complexity_grouped = complexity_raw.groupby('sequence_length')
    complexity = complexity_grouped.mean().rename(columns={'operations_number': 'operations_number_mean',
                                                           'timers': 'timers_mean'})
    complexity['operations_number_std'] = complexity_grouped['operations_number'].std()
    complexity['timers_std'] = complexity_grouped['timers'].std()
    petri_net_size = len(petri_net.arcs) + len(petri_net.places) + len(petri_net.transitions)
    complexity['worst_case'] = complexity.index*complexity.index*(petri_net_size)
    complexity =  complexity.fillna(0)
    complexity.to_csv(f'./results/complexity-{net_name}.csv')
    breakpoint()
    #training_data = algorithm.old_extract_decision_points_data_only_last_event(log)
    # Data has been gathered. For each decision point, fitting a decision tree on its logs and extracting the rules
    train(training_data, attributes_map, net_name, './models')
    toc = time()
    print("\nTotal time: {}".format(toc-tic))


if __name__ == "__main__":
    main()
