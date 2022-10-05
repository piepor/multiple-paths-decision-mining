def get_attributes_from_event(event) -> dict:
    """ Given an event from an event log, returns a dictionary containing the attributes values of the given event.

    Each attribute is set as key of the dictionary, and its value is the actual value of the attribute in the event.
    Attributes 'time:timestamp' and 'concept:name' are ignored and therefore not added to the dictionary.
    """

    attributes = dict()
    for attribute in event.keys():
        if attribute not in ['time:timestamp', 'concept:name']:
            attributes[attribute] = event[attribute]

    return attributes


def get_map_events_to_transitions(net) -> dict:
    """ Compute a mapping between log events and Petri Net transitions names.

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every event the corresponding transition name.
    """
    # initialize
    map_trans_events = dict()
    for trans in net.transitions:
        map_trans_events[trans.label] = trans.name
    map_trans_events['None'] = 'None'
    return map_trans_events


def get_map_transitions_to_events(net) -> dict:
    """ Compute a mapping between Petri Net transitions names and log events.

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every transition the corresponding event name.
    """
    # initialize
    map_events_trans = dict()
    for trans in net.transitions:
        if trans.label is not None:
            map_events_trans[trans.name] = trans.label
    return map_events_trans
