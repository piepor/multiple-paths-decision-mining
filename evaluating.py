from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog, Trace

def evaluate_data_petri_net(petri_net: PetriNet, event_log: EventLog):
    for trace in event_log:
        metric = evaluate_trace(petri_net, trace)
    return metrics

def evaluate_trace(petri_net: PetriNet, trace: Trace):
    attributes = {}
    for event in trace:
        attributes.update
