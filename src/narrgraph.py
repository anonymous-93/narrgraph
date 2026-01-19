import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GATv2Conv
from torch.nn import BatchNorm1d
from transformers import BertModel, AutoConfig, BertTokenizer
from torch_geometric.nn import BatchNorm
import random
import torch
import tokenizations
import numpy as np

def set_all_seeds(seed=42):
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Optional: environment variable for further determinism
    os.environ['PYTHONHASHSEED'] = str(seed)




class NarrGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, config, dep_size, relation_type_mapping, iob_labels):
        super(NarrGraph, self).__init__()
        decoder_size = 512
        dep_size = dep_size + 3 #seq, full
        self.num_entity_labels = len(iob_labels)
        num_relation_labels = len(relation_type_mapping)

        # Encoder Layers
        self.bert = BertModel(config, add_pooling_layer=False)
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4, concat=False, edge_dim=dep_size)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False,  edge_dim=dep_size)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False,  edge_dim=dep_size)

        # Latent Layers (GVAE)
        #self.conv_mu = TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False,  edge_dim=dep_size)
        #self.conv_logstd = TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False,  edge_dim=dep_size)


        # Decoder layers
        self.decoder_dense_1 = nn.Linear(hidden_dim*2 + self.num_entity_labels*2, decoder_size)
        self.decoder_dense_2 = nn.Linear(decoder_size, decoder_size)
        self.decoder_dense_3 = nn.Linear(decoder_size, decoder_size)
        self.label_pred_layer = nn.Linear(decoder_size, len(relation_type_mapping)) #srlinks and tlinks | qslinks
        self.entity_classifier = nn.Linear(hidden_dim, self.num_entity_labels)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
  # Relation classifier dropout
        self.relation_dropout = nn.Dropout(classifier_dropout)


    def encode(self, x, edge_index, edge_attr):
        h1 = self.conv1(x, edge_index, edge_attr)
        #h1 = F.leaky_relu(h1)
        #h2 = self.conv2(h1, edge_index, edge_attr) + h1
        #h2 = F.leaky_relu(h2)
        #h3 = self.conv3(h2, edge_index, edge_attr) + h2
        return h1


  
    def preprocess_bert(self, data):
        encodings = data.encodings
        bert_output = self.bert(encodings["input_ids"], attention_mask=encodings["attention_mask"], token_type_ids=encodings["token_type_ids"])
        #return bert_output[0]
        bert_embeddings = bert_output.last_hidden_state#.squeeze()
        all_embeddings = []
        #print("BERT embeddings shape:", bert_embeddings.shape)
        all_embeddings = []
        for i, indices_list in enumerate(encodings['alignment']):
            z_embeddings = []
            for j, indices in enumerate(indices_list):
                if indices:
                    token_embedding = bert_embeddings[i, indices[0], :]
                else:
                    token_embedding = torch.zeros(bert_embeddings.shape[2]).to(bert_embeddings.device)

                # Concatenate POS one-hot vector
                pos_vec = encodings['pos_vectors'][j].to(token_embedding.device)
                token_embedding = torch.cat([token_embedding, pos_vec], dim=-1)

                z_embeddings.append(token_embedding)

            all_embeddings.append(torch.stack(z_embeddings))
        
        return torch.cat(all_embeddings, dim=0)
    
    
    def decode(self, z):
        #print(z.shape)
        graph_z = z
        #print("Graph Z shape:", graph_z.shape)
        edge_indicesX = torch.triu_indices(graph_z.shape[0], graph_z.shape[0], offset=1) # calculate upper part of the matrix indexes
        #print("Edge indicesX shape:", edge_indicesX.shape)
        source_indices = edge_indicesX[0]
        #print(source_indices.shape)
        target_indices = edge_indicesX[1]
        #print("im i alive?")
        source_features = graph_z[source_indices]
        target_features = graph_z[target_indices]

        graph_inputs = torch.cat([source_features, target_features], axis=1)
        # Get predictions
        x = self.decoder_dense_1(graph_inputs).relu()
        x = self.decoder_dense_2(x).relu()
        #x = self.decoder_dense_3(x).relu()
        #print("fiimm")
        x = self.relation_dropout(x)
        edge_logits = self.label_pred_layer(x)
        return  edge_logits #edge_indicesX #graph_inputs


        
    def reparameterize(self, mu, logstd):
        if self.training:
            #getting the std deviation
            std = torch.exp(logstd)
            #generate a random number from a normal distribution
            eps = torch.randn_like(std)
            #return the z value
            return eps.mul(std).add_(mu)
        else:
            #return the mean
            return mu

    def forward(self, data, pipeline=False):
        data = data.to(device)
        #print(data)
        x , edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.preprocess_bert(data)
        #print(x.shape)
        z = self.encode(x, edge_index, edge_attr) #mu, logstd
        #print(z.shape)
        sequence_output = self.dropout(z)
        entity_logits_init = self.entity_classifier(sequence_output)


        if self.training:

            entity_features_detached = entity_logits_init.detach()
        else:
            if pipeline:
                entity_features_detached = data.entity_preds.to(device)
                print("using predicted entity labels")
            else:
                entity_features_detached = entity_logits_init.detach()
        z = torch.cat([z, entity_features_detached], dim=1) #data.label_one_hot # entity_one_hot
        
        edge_logits = self.decode(z) #data.label_one_hot

        return entity_logits_init, edge_logits
