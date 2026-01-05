import preprocess_data
import model_handler
import os
import re
import spacy





test_set_ids = ['lusa_97',
 'lusa_4',
 'lusa_67',
 'lusa_20',
 'lusa_83',
 'lusa_104',
 'lusa_80',
 'lusa_79',
 'lusa_34',
 'lusa_47',
 'lusa_30',
 'lusa_96',
 'lusa_11',
 'lusa_112',
 'lusa_100',
 'lusa_77',
 'lusa_38',
 'lusa_86',
 'lusa_60']


val_set_ids = ['lusa_12',
 'lusa_48',
 'lusa_45',
 'lusa_115',
 'lusa_44',
 'lusa_82',
 'lusa_111',
 'lusa_63',
 'lusa_8',
 'lusa_42',
 'lusa_50',
 'lusa_76',
 'lusa_114',
 'lusa_99',
 'lusa_18']


entity_labels = ["O", "Event", "Participant", "Time", "Spatial_Relation"] #,"Time","Spatial_Relation"]
iob_labels = ["O", "B-Event", "I-Event", "B-Participant", "I-Participant", "B-Time", "I-Time", "B-Spatial_Relation", "I-Spatial_Relation"]
label2id = {label: i  for i, label in enumerate(entity_labels)}
iob2id = {label: i  for i, label in enumerate(iob_labels)}
id2label = {i: label for label, i in label2id.items()}
id2iob = {i: label for label, i in iob2id.items()}
universal_pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET",
                      "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                      "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
relation_type_mapping = {'TLINK': 0, 'SRLINK': 1, 'QSLINK': 2, "OLINK": 3}


if __name__ == "__main__":
    path = "./data"
    nlp = spacy.load('pt_core_news_lg')
    
    depsize = len(nlp.get_pipe("parser").labels)
    
    files_train = []
    files_test = []
    files_val = []

    for file in os.listdir(path):
        if file.endswith(".txt") and file.startswith("lusa") and file: #not in exceptions:
            file = re.match(r'(.*)\.txt',file)
            if file[1] in test_set_ids:
                files_test.append(os.path.join(path, file[1]))
            elif file[1] in val_set_ids:
                files_val.append(os.path.join(path, file[1]))
            else:
                files_train.append(os.path.join(path, file[1]))
                
    train_loader, test_loader, val_loader = preprocess_data.generateDataLoaders(files_train, files_val, files_test, relation_type_mapping, nlp)
    model_handler.train_model(train_loader, val_loader, test_loader, universal_pos_tags, iob_labels, iob2id, id2iob, relation_type_mapping, depsize)

    
    