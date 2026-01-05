from brat_parser import get_entities_relations_attributes_groups
import os
import re

def parseBratFile(file):
    entities, relations, attributes, groups = get_entities_relations_attributes_groups(file)
    #filter movelink relations
    relations = {id: rel for id, rel in relations.items() if "MOVELINK" not in rel.type}
    return entities, relations, attributes, groups
# The praser returns dataclasses.


def filter_annotations(annotations, type, match=False, exceptions=[]):
    if match:
        return {id: ann for id, ann in annotations.items() if type in ann.type and ann.type not in exceptions}
    else:
        return {id: ann for id, ann in annotations.items() if ann.type == type and ann.type not in exceptions}



def filter_relation_by_entity(relations, entities, link_type_prefix, entity_types={"Event", "Time"}):
    filtered = {}
    discarded = {}
    for rid, rel in relations.items():
        # Only TLINK relations
        if not rel.type.startswith(link_type_prefix):
            continue

        subj_ent = entities.get(rel.subj)
        obj_ent = entities.get(rel.obj)

        # Skip if entities are missing
        if subj_ent is None or obj_ent is None:
            continue

        # Event–Time or Time–Event
        types = {subj_ent.type, obj_ent.type}
        if types == entity_types:
            rel.type = "event-" + rel.type
            filtered[rid] = rel
            
        else:
            discarded[rid] = rel

    return filtered, discarded

def parse_brat(files):
    entities_list = []
    relations_list = []
    texts_list = []
    for file in files:
        #file = "../../Datasets/lusa_news_final/lusa_1"
        entities, relations, attributes, _ = parseBratFile(file + ".ann")
        #tlinks_event, tlinks_other = filter_relation_by_entity(relations, entities, "TLINK_", entity_types={"Event", "Time"})
        events = filter_annotations(entities, "Event")
        times = filter_annotations(entities, "Time")
        spatial_relations = filter_annotations(entities, "Spatial_Relation")
        participants = filter_annotations(entities, "Participant")
        #entities_ = events | participants | times | spatial_relations
        entities_ = events | participants | times | spatial_relations
        tlinks =  filter_annotations(relations, "TLINK", match=True)
        qslinks =  filter_annotations(relations, "QSLINK", match=True)
        olink =  filter_annotations(relations, "OLINK", match=True)#, exceptions=["OLINK_objIdentity"])
        mlink =  filter_annotations(relations, "MOVELINK", True)
        srl_links = filter_annotations(relations, "SRL", match=True)
        #relations_ = tlinks | qslinks | olink | mlink | srl_links
        relations_ = srl_links|  qslinks | olink | tlinks #tlinks_event | tlinks_other #tlinks #
        text = open(file + ".txt").read()
        entities_list.append(entities_)
        relations_list.append(relations_)
        texts_list.append(text)
    return entities_list, relations_list, texts_list
