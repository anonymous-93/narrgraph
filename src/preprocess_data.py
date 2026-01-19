import spacy 
import torch
import os 
import re
from parse_brat import parse_brat






## geenrate node features 

from transformers import BertTokenizerFast
import tokenizations



entity_labels = ["O", "Event", "Participant", "Time", "Spatial_Relation"] #,"Time","Spatial_Relation"]
iob_labels = ["O", "B-Event", "I-Event", "B-Participant", "I-Participant", "B-Time", "I-Time", "B-Spatial_Relation", "I-Spatial_Relation"]
label2id = {label: i  for i, label in enumerate(entity_labels)}
iob2id = {label: i  for i, label in enumerate(iob_labels)}
id2label = {i: label for label, i in label2id.items()}
id2iob = {i: label for label, i in iob2id.items()}
universal_pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET",
                      "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                      "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]


#One Hot Encoding para labels de entidades / dep_parser labels
def one_hot_encode(index, size):
    one_hot = torch.zeros(size, dtype=torch.float)
    one_hot[index] = 1.0
    return one_hot

def get_spacy_token_indices(span, doc, idx):
        start, end = span[0], span[1]
        #print(start, end)
        
        token_indices = []
        for token in doc:
            #print(token)
            if token.idx >= start and token.idx + len(token.text) <= end: # rever...
                token_indices.append(token.i)
            elif token.idx >= start and token.idx <= end <= token.idx + len(token.text):
                token_indices.append(token.i)
            elif token.idx <= start  <= token.idx + len(token.text) and token.idx + len(token.text) <= end:
                token_indices.append(token.i)
        #if token_indices == []:
            
            #print("Error when mapping spacy tokens to entities spans")
            #print(span)
            #print(doc.text[start:end])
            #print(doc.text[start- 10:end+10])
            #print(doc)
            #print("Entity ID:", idx)
            #for token in doc:
                #print(token.idx, len(token.text))
                #print(token.text)
        #if idx == "T56":
        #    print("T56", span, start, end, token_indices)
        ##    print("T56", token_indices)
        #    token = doc[token_indices[0]]
        #    print("T56", token.idx, token.text)
        return token_indices

# Calcular a média dos embeddings dos tokens de uma entidade
def aggregate_embeddings(indices, embeddings):
    if indices:
        
        return torch.mean(embeddings[indices], dim=0)
    else:
        print("error calculating entity embeddings")
        #print(indices)
        #print(entity_indices)
        return torch.zeros(embeddings.size(1))



def entity_indices_to_iob(n_tokens, entity_indices):
    iob_tags = [iob2id['O']] * n_tokens  # default all outside

    for token_indices, ent_type_idx, _ in entity_indices:
        ent_type = id2label[ent_type_idx]
        if ent_type != "O":
            for i, token_idx in enumerate(token_indices):
                prefix = 'B' if i == 0 else 'I'
                iob = f"{prefix}-{ent_type}"
                iob_tags[token_idx] = iob2id[iob]
    return iob_tags





def generateEntityLabels(doc, entity_indices):
    n_tokens = len(doc)
    labels = entity_indices_to_iob(n_tokens, entity_indices)
    # generate one-hot encoding for entity labels
    entity_type_one_hot = torch.stack([one_hot_encode(label, len(iob_labels)) for label in labels])
    return labels, entity_type_one_hot


tokenizer = BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased') ## neuralmind/bert-base-portuguese-cased

def tokenize_and_align_labels(tokenized_inputs, labels):
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
        # ignored in the loss function.
        if word_idx is None:
            label_ids.append(-100)
        # We set the label for the first token of each word.
        elif word_idx != previous_word_idx:
            label_ids.append(labels[word_idx])
        # For the other tokens in a word, we set the label to either the current label or -100, depending on
        # the label_all_tokens flag.
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = torch.tensor([label_ids])
    return tokenized_inputs        

def generate_bert_encodings(doc, labels):
    tokens = [token.text for token in doc]
    encodings = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', max_length=512, truncation=True, padding='max_length', add_special_tokens=True)
    encodings = tokenize_and_align_labels(encodings,labels)
    
    tokens_b = tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])
    tokens_b = [tok for tok in tokens_b if tok != tokenizer.pad_token]
    a2b, _ = tokenizations.get_alignments(tokens, tokens_b)
    encodings['alignment'] = a2b
    encodings['pos_vectors'] = [one_hot_encode(universal_pos_tags.index(token.pos_), len(universal_pos_tags)) for token in doc]
    return encodings

   
   
## #### Generate input Edge_Index with the dependency parser tree (graph connections)


def accumulate_one_hot(edge_dict, edge, one_hot_attr):
    if edge not in edge_dict:
        edge_dict[edge] = one_hot_attr
    else:
        edge_dict[edge] = torch.logical_or(edge_dict[edge].bool(), one_hot_attr.bool()).float() # Accumulate one-hot vectors
    return edge_dict


def generateEdgeIndex(doc, input_rel_label2id, strategy=["dep", "seq"]):
    # Create the dependency graph between entities
    edge_index = []
    edge_attr = []
    unique_edges = []
    n_tokens = len(doc)
    edges_dict = {}
    #token_to_entity_idx = generateTokenToEntityIndex(entity_indices)
    if "dep" in strategy:  
        for token in doc:
            if token.head != token: #and token.i in token_to_entity_idx and token.head.i in token_to_entity_idx: # se o token e o head forem entidades
                edge = (token.i, token.head.i)
                #reverse = (token_to_entity_idx[token.head.i], token_to_entity_idx[token.i])
                one_hot_attr = one_hot_encode(input_rel_label2id[token.dep_], len(input_rel_label2id)) #
                if edge[0] != edge[1]:
                    edges_dict = accumulate_one_hot(edges_dict, edge, one_hot_attr)
            
    if "seq" in strategy:
        for i in range(n_tokens - 1):
            edge = (i, i + 1)
            one_hot_attr = one_hot_encode(input_rel_label2id["seq"], len(input_rel_label2id ))
            edges_dict = accumulate_one_hot(edges_dict, edge, one_hot_attr)

    if "full" in strategy:
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                edge = (i, j)
                one_hot_attr = one_hot_encode(input_rel_label2id["full"], len(input_rel_label2id))
                edges_dict = accumulate_one_hot(edges_dict, edge, one_hot_attr)
    
    edge_index = torch.tensor(list(edges_dict.keys()), dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(list(edges_dict.values()))

    #print("Edge index:", edge_index.shape)
    return edge_index, edge_attr
    
    
    

### Generate output Output Edge_Index with LUSA relations



k = []
def generateTargetEdgeIndex(relations, node_id_mapping, relation_type_mapping):
    edge_indices = []
    edge_dict = {}
    edge_attr_text_dict = {} 
    global k
    edge_attributes = []
    edge_attr_text = []
    # generate edge
    #print(relations.values())
    #print(node_id_mapping)
    for rel in relations.values():
        #print(rel)
        if rel.subj in node_id_mapping and rel.obj in node_id_mapping:
            subj_idx = node_id_mapping[rel.subj]
            obj_idx = node_id_mapping[rel.obj]
            #if subj_idx == obj_idx:
                #print("ERROR: Self-loop detected")
                #print(rel)
            edge_key = tuple(sorted([subj_idx, obj_idx]))
            #one_hot_attr = torch.tensor(one_hot_encode(relation_type_mapping[rel.type.split("_")[0]],len(relation_type_mapping))).clone().detach()
            one_hot_attr = torch.as_tensor(one_hot_encode(relation_type_mapping[rel.type.split("_")[0]], len(relation_type_mapping))).clone().detach() #rel.type.split("_")[0]

   
                # If the edge exists, sum the one-hot encoded attributes
            if edge_key in edge_dict:
                
                edge_dict[edge_key] = torch.logical_or(edge_dict[edge_key].bool(), one_hot_attr.bool()).float() # Accumulate one-hot vectors 
                edge_attr_text_dict[edge_key].add(rel.type.split("_")[0])  # Store unique text labels
            else:
                edge_dict[edge_key] = one_hot_attr
                edge_attr_text_dict[edge_key] = {rel.type.split("_")[0]}
        else:
            print(rel)
            print("ERROR: Relation refers to missing nodes")
    # Convert the dictionary into tensors
# Convert the dictionary into tensors
    if len(edge_dict) > 0:
        edge_indices_y = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t().contiguous()
        edge_attr_y = torch.stack(list(edge_dict.values()))
    else:
        edge_indices_y = torch.empty((2, 0), dtype=torch.long)
        edge_attr_y = torch.empty((0, len(relation_type_mapping)), dtype=torch.float)

    edge_attr_text = ["|".join(sorted(types)) for types in edge_attr_text_dict.values()]
    
    return edge_indices_y, edge_attr_y, edge_attr_text




### Generate a input and output graph given a Lusa document


import json
def get_entity_preds(filename):
    entity_preds = json.load(open(filename))
    #one_hot_encode entity preds
    entity_preds_one_hot = []
    for batch in entity_preds:
        batch_preds = []
        for pred in batch:
            batch_preds.append(one_hot_encode(iob2id[pred], len(iob_labels)))
        entity_preds_one_hot.append(batch_preds)
    return entity_preds_one_hot



from torch_geometric.data import Data

# Create a mapping from token indices to entity indices
def generateTokenToEntityIndex(entity_indices):
    token_to_entity_idx = {}
    entity_sizes = {i: len(indices) for i, (indices, _, _) in enumerate(entity_indices)}
    discarded_entities = set()
    for entity_idx, (indices, type, idx) in enumerate(entity_indices):
        for token_idx in indices:
            #print(token_idx)
            # If token already assigned, keep the entity with more tokens
            if token_idx in token_to_entity_idx:
                current_entity = token_to_entity_idx[token_idx]
                if entity_sizes[entity_idx] > entity_sizes[current_entity]:
                    discarded_entities.add(entity_indices[current_entity][2])  # Track discarded entity
                    token_to_entity_idx[token_idx] = entity_idx
                else:
                    discarded_entities.add(idx)  # Track the smaller entity
            else:
                token_to_entity_idx[token_idx] = entity_idx
    return token_to_entity_idx, discarded_entities

def getGroupedTokens(doc, entity_indices):
    grouped_tokens = []
    visited_tokens = set()	
    indices_entity, discarded_entities = generateTokenToEntityIndex(entity_indices)
    for token in doc:
        if token.i not in indices_entity:
            grouped_tokens.append(([token.i], 0, "")) # out of entity index: 0
        else:
            entity_idx = indices_entity[token.i]
            if entity_idx not in visited_tokens:
                grouped_tokens.append(entity_indices[entity_idx])
                visited_tokens.add(entity_idx)
    return grouped_tokens, discarded_entities

def generateDataset(entities, relations, text, relation_type_mapping, input_rel_label2id, nlp):
    text = text.replace("\n", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("´", "'")
    doc = nlp(text)

    entity_indices = [(get_spacy_token_indices(entity.span[0], doc, idx), label2id[entity.type], idx) for idx, entity in entities.items()]
    #print(entity_indices)
    
    labels, label_one_hot = generateEntityLabels(doc, entity_indices)
    tokens = [token.text for token in doc]
    encodings = generate_bert_encodings(doc,labels)
    #print("len(tokens):", len(tokens))
    #print("len(encodings):", len(encodings['input_ids']))
    #print("len (entity_indices):", len(entity_indices))
    # Get edge_index 
    iob_labels = [id2iob[label] for label in labels]
    #print("len(labels_iob):", len(iob_labels))
    edge_index, edge_attr = generateEdgeIndex(doc, input_rel_label2id)
 
    node_id_mapping = {node_id: indices[0] for idx, (indices,type,node_id) in enumerate(entity_indices)}
    y_edge_index, y_edge_attr, edge_attr_text = generateTargetEdgeIndex(relations, node_id_mapping, relation_type_mapping)

    #print("Entity text:", tokens)
    data = Data(
        x=torch.zeros(len(tokens)), 
        edge_index=edge_index, edge_attr=edge_attr, 
        y=torch.tensor(labels), 
        encodings = encodings, 
        target_edge_index=y_edge_index, 
        y_edge_attr=y_edge_attr, 
        tokens= tokens,
        labels = torch.tensor(labels), 
        iob_labels = iob_labels, 
        label_one_hot = torch.tensor(label_one_hot),
    )
    return data



#relation_type_mapping = relation_label2id
def generateDatasets(files, relation_type_mapping, nlp):
    data_list = []
    entities_list, relations_list, texts_list = parse_brat(files)
    #print(len(entities_list), len(relations_list), len(texts_list))
    
    #relation_types = ['TLINK', 'SRLINK', 'QSLINK'] #list(set(rel.type.split("_")[0] for relations in relations_list for rel in relations.values()))
    #print(relation_types)
    
    #print(relation_type_mapping)
    input_rel_label2id = {label: idx for idx, label in enumerate(list(nlp.get_pipe("parser").labels) + ["seq", "full", "entity"])}
    #print(len(relation_type_mapping), len(dep_label_mapping))
    for i, entities in enumerate(entities_list):
        #print(files[i])
        #if relations_list[i] == {}:
            #print(texts_list[i])
            #print(i)
        data = generateDataset(entities, relations_list[i], texts_list[i], relation_type_mapping,input_rel_label2id, nlp)
        data['file_name'] = files[i]
        #print(data)
        #print(i)
        data_list.append(data)
        #loader = DataLoader(data_list, batch_size=32)
    return data_list

\
from torch_geometric.loader import DataLoader

def generateDataLoaders(files_train, files_val, files_test,relation_type_mapping,nlp, batch_size=1):
    data_list_train = generateDatasets(files_train, relation_type_mapping, nlp)
    data_list_test = generateDatasets(files_test, relation_type_mapping, nlp)
    data_list_val = generateDatasets(files_val, relation_type_mapping, nlp)
    entity_preds = get_entity_preds("lusa_ner_predictions_test.json")
    
    for data, preds in zip(data_list_test, entity_preds):
        data.entity_preds = torch.stack(preds)
    entity_preds = get_entity_preds("lusa_ner_predictions_val.json")
    for data, preds in zip(data_list_val, entity_preds):
        data.entity_preds = torch.stack(preds)

    train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(data_list_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data_list_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader