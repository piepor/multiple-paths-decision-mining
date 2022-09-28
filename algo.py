from tqdm import tqdm
from typing import Union
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet
from backward_search import get_decision_points_and_targets, get_all_dp_from_sink_to_last_event
from model_utils import get_transition, get_sink


class MultiplePaths:
    def __init__(self, petri_net: PetriNet, activities_to_trans_map: dict):
        self.petri_net = petri_net
        self.activities_to_trans_map = activities_to_trans_map
        self.sink = get_sink(self.petri_net)
        self.stored_dicts = dict()
        self.dp_dict = dict()

    def activities_decision_points_map(self, variant: str) -> dict:
        transitions_sequence, activities = list(), list()
        dp_activities = dict()
        for i, activity in enumerate(variant.split(',')):
            trans_from_activity = self.activities_to_trans_map[activity]
            transitions_sequence.append(trans_from_activity)
            activities.append(activity)
            if len(transitions_sequence) > 1:
                self.dp_dict, self.stored_dicts = get_decision_points_and_targets(transitions_sequence, self.petri_net, self.stored_dicts)
                dp_activities['Activity_{}'.format(i+1)] = self.dp_dict
        # Final update of the current trace (from last event to snk)
        transition =  get_transition(self.petri_net, activity)
        dp_activities['End'] = get_all_dp_from_sink_to_last_event(transition, self.sink, dp_activities)
        return dp_activities

    def extract_decision_points_data(self, event_log: EventLog) -> dict:
        decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
        variants = variants_filter.get_variants(event_log)
        # Decision points of interest are searched considering the variants only
        for variant in tqdm(variants):
            dp_events_sequence = activities_decision_points_map(variant, events_to_trans_map, petri_net, stored_dicts)
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
