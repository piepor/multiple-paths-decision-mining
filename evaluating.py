from itertools import count
import numpy as np
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking
from pm4py.algo.conformance.alignments.petri_net import algorithm as pn_align
from pm4py.objects.petri_net.importer import importer as pnml_importer

def evaluate_data_petri_net(petri_net: PetriNet, event_log: EventLog, decision_points: dict, initial_marking: Marking, final_marking: Marking):
    # returns also the name of the transitions
    dps_map = get_dps(petri_net)
    parameters = {pn_align.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    aligned_traces = pn_align.apply_log(event_log, petri_net, initial_marking, final_marking, parameters=parameters)
    metrics = []
    for i, alignment in enumerate(aligned_traces):
        trace = event_log[i]
        metrics.append(evaluate_trace(alignment, trace, decision_points, dps_map))
    return sum(metrics) / len(event_log)

def evaluate_trace(alignment: dict, trace: Trace, decision_points: dict, dps_map: dict) -> float:
    breakpoint()
    transitions = []
    prob_sequence = []
    for align in alignment['alignment']:
        trans = align[0][1]
        # TODO add the case of a transitions pointed by two dps
        if trans in dps_map:
            attributes = get_attributes(transitions, trace)
            dp = dps_map[trans][0]
            #classifier = decision_points[dp]
            #trans_prob = classifier.predict_prob(trans, attributes)
            prob_sequence.append(np.log(0.5))
        transitions.append(align[1][1])
    breakpoint()
    return np.exp(sum(prob_sequence))

def get_attributes(trans: list, trace: Trace):
    if not trans:
        return []
    attributes = {}
    visible_trans = [transition for transition in trans if not transition is None]
    last_trans = visible_trans[-1]
    count_trans = trans.count(last_trans)
    trace_seq = [event['concept:name'] for event in trace]
    occurrences = [ind for ind, ele in zip(count(), trace_seq) if ele == last_trans]
    if not occurrences:
        return []
    index_trans = occurrences[count_trans-1]
    for i in range(index_trans+1):
        event = trace[i]
        for event_attr in event.keys():
            attributes[event_attr] = event[event_attr]
    return attributes
        
def get_dps(petri_net: PetriNet):
    dps = {}
    for place in petri_net.places:
        if len(place.out_arcs) > 1:
            for arc in place.out_arcs:
                trans = arc.target.name
                if not trans in dps.keys():
                    dps[trans] = []
                dps[trans].append(place.name)
    return dps


log = pm4py.read_xes('logs/log-Road_Traffic_Fine_Management_Process.xes')
net, im, fm = pnml_importer.apply('models/Road_Traffic_Fine_Management_Process.pnml')
evaluate_data_petri_net(net, log, {}, im, fm)
