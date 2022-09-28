import argparse
from time import time
from pm4py.objects.log.importer.xes import importer as xes_importer
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier
from importing import import_attributes, convert_attributes, import_petri_net
from model_utils import get_sink, activities_to_transitions_map


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
    activities_to_trans_map = activities_to_transitions_map(petri_net)
    # TODO declare algo type

    tic = time()
    # Scanning the log to get the logs related to decision points
    print('Extracting training logs from Event Log...')
    decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
    variants = variants_filter.get_variants(log)
    # Decision points of interest are searched considering the variants only
    for variant in tqdm(variants):
        transitions_sequence, events_sequence = list(), list()
        dp_events_sequence = dict()
        for i, event_name in enumerate(variant.split(',')):
            trans_from_event = events_to_trans_map[event_name]
            transitions_sequence.append(trans_from_event)
            events_sequence.append(event_name)
            if len(transitions_sequence) > 1:
                dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, net, stored_dicts)
                dp_events_sequence['Event_{}'.format(i+1)] = dp_dict

        # Final update of the current trace (from last event to sink)
        transition = [trans for trans in net.transitions if trans.label == event_name][0]
        dp_events_sequence['End'] = get_all_dp_from_sink_to_last_event(transition, sink_complete_net, dp_events_sequence)

        for trace in variants[variant]:
            # Storing the trace attributes (if any)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                event_attr.update(trace.attributes)

            # Keeping the same attributes observed in previously (to keep dictionaries at the same length)
            event_attr = {k: np.nan for k in event_attr.keys()}

            transitions_sequence = list()
            for i, event in enumerate(trace):
                trans_from_event = events_to_trans_map[event["concept:name"]]
                transitions_sequence.append(trans_from_event)

                # Appending the last attribute values to the decision point dictionary
                if len(transitions_sequence) > 1:
                    dp_dict = dp_events_sequence['Event_{}'.format(i+1)]
                    for dp in dp_dict.keys():
                        # Adding the decision point to the total dictionary if it is not already there
                        if dp not in decision_points_data.keys():
                            decision_points_data[dp] = {k: [] for k in ['target']}
                        for dp_target in dp_dict[dp]:
                            for a in event_attr.keys():
                                # Attribute not present and not nan: add it as new key and fill previous entries as nan
                                if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                    n_entries = len(decision_points_data[dp]['target'])
                                    decision_points_data[dp][a] = [np.nan] * n_entries
                                    decision_points_data[dp][a].append(event_attr[a])
                                # Attribute present: just append it to the existing list
                                elif a in decision_points_data[dp]:
                                    decision_points_data[dp][a].append(event_attr[a])
                            # Appending also the target transition label to the decision point dictionary
                            decision_points_data[dp]['target'].append(dp_target)

                # Updating the attribute values dictionary with the values from the current event
                event_attr.update(get_attributes_from_event(event))

            # Appending the last attribute values to the decision point dictionary (from last event to sink)
            if len(dp_events_sequence['End']) > 0:
                for dp in dp_events_sequence['End'].keys():
                    if dp not in decision_points_data.keys():
                        decision_points_data[dp] = {k: [] for k in ['target']}
                    for dp_target in dp_events_sequence['End'][dp]:
                        for a in event_attr.keys():
                            if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                n_entries = len(decision_points_data[dp]['target'])
                                decision_points_data[dp][a] = [np.nan] * n_entries
                                decision_points_data[dp][a].append(event_attr[a])
                            elif a in decision_points_data[dp]:
                                decision_points_data[dp][a].append(event_attr[a])
                        decision_points_data[dp]['target'].append(dp_target)

if __name__ == "__main__":
    main()
