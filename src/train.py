import torch

from torch_geometric.utils import to_dense_adj
 # N(N-1)/2
def edge_index_2_adj_matrix_label(data):
    #print(data)
    matrix = torch.squeeze(to_dense_adj(data.target_edge_index, edge_attr=data.y_edge_attr, max_num_nodes=data.x.shape[0]))
    triu_indices = torch.triu_indices(matrix.shape[0], matrix.shape[0], offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    #print(edge_logits.squeeze().shape, batch_targets[triu_mask].shape)
    
    return matrix[triu_mask]

def edge_index_2_adj_matrix(data):
    #print(data)
    batch_targets = torch.squeeze(to_dense_adj(data.y, max_num_nodes=data.x.shape[0]))
    triu_indices = torch.triu_indices(batch_targets.shape[0], batch_targets.shape[0], offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    #print(edge_logits.squeeze().shape, batch_targets[triu_mask].shape)
    return batch_targets[triu_mask]

# Kullback–Leibler divergence
def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd =  logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div

    
# Training loop
def train(model, data_list, label_loss_fn, entity_loss_fn, optimizer,iob_labels, device, scheduler = None):
    model.train()
    for i, data in enumerate(data_list):
        #print("new batch", i)
        data.to(device)
        #print(i)
        optimizer.zero_grad()
        entity_logits_init, edge_logits_joint = model(data) #, label_logits_final
        edge_labels_y = edge_index_2_adj_matrix_label(data)
        edge_loss_joint = label_loss_fn(edge_logits_joint, edge_labels_y)
        entity_loss_init = entity_loss_fn(entity_logits_init.view(-1, len(iob_labels)), data.labels.view(-1))
        total_loss = edge_loss_joint + entity_loss_init * 0.1 #+ edge_loss_joint   #entity_loss_final*0.1 #+ kl_divergence
        total_loss.backward()
        optimizer.step()
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
    return total_loss
#train(model, data_list_train, criterion, optimizer)






import evaluate
metric = evaluate.load("seqeval")
from tqdm import tqdm
import numpy as np
from seqeval.metrics import classification_report
#from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import (
    f1_score as sklearn_f1,
    precision_score as sklearn_p,
    recall_score as sklearn_r
)
def evaluate(model, eval_dataloader, device, id2iob, relation_type_mapping, pipeline=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    edge_all_y = []
    edge_all_probs = []
    edge_all_probs_joint = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            entity_logits, edge_logits_joint = model(batch, pipeline=pipeline)
            predictions = torch.argmax(entity_logits, dim=-1)
            entity_labels = batch.labels
            
            edge_y = edge_index_2_adj_matrix_label(batch)  # assume tensor
            #edge_probs = torch.sigmoid(edge_logits)       # tensor
            edge_probs_joint = torch.sigmoid(edge_logits_joint) # tensor

            # Accumulate
            edge_all_y.extend(edge_y.cpu())         # move to CPU 
            #edge_all_probs.extend(edge_probs.cpu()) # move to CPU 
            edge_all_probs_joint.extend(edge_probs_joint.cpu()) # move to CPU
            
            entity_labels = [id2iob[label.item()] for label in entity_labels]  # Convert to IOB format
            predictions = [id2iob[pred.item()] for pred in predictions]  # Convert to IOB format
            all_preds.append(predictions)
            all_labels.append(entity_labels)

    # Now concatenate tensors
    edge_all_y = np.array(edge_all_y)
    edge_all_probs_joint = np.array(edge_all_probs_joint)
    edge_all_preds_joint = (edge_all_probs_joint > 0.5).astype(int)
    

    # Compute binary predictions
    #edge_f1 = sklearn_f1(edge_all_y, edge_all_preds, average="micro", zero_division=0)
    edge_joint_f1 = sklearn_f1(edge_all_y, edge_all_preds_joint, average="micro", zero_division=0)
    edge_joint_recall = sklearn_r(edge_all_y, edge_all_preds_joint, average="micro", zero_division=0)
    edge_joint_precision = sklearn_p(edge_all_y, edge_all_preds_joint, average="micro", zero_division=0)
    
    
    edge_precision = sklearn_p(edge_all_y, edge_all_preds_joint, average=None, zero_division=0)
    edge_recall = sklearn_r(edge_all_y, edge_all_preds_joint, average=None, zero_division=0)
    edge_f1 = sklearn_f1(edge_all_y, edge_all_preds_joint, average=None, zero_division=0)
    
    
    num_instances_per_class = np.sum(edge_all_y, axis=0)  # True labels
    num_preds_per_class = np.sum(edge_all_preds_joint, axis=0)  # Predicted labels

    inverted_mapping = {v: k for k, v in relation_type_mapping.items()}
    print("Label Metrics by Class:")
    for class_id, (precision, recall, f1) in enumerate(zip(edge_precision, edge_recall, edge_f1)):
        support = num_instances_per_class[class_id]
        predictions = num_preds_per_class[class_id]
        print(f"Class {inverted_mapping[class_id]}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Support (true instances):     {int(support)}")
        print(f"  Predicted instances:          {int(predictions)}")
        print("-" * 30)
    

    f1 = metric.compute(predictions=all_preds, references=all_labels)["overall_f1"]



    print("\nEntity metrics by label:")
    print(classification_report(
        all_labels,
        all_preds,
        digits=4
    ))
    print(f"Edge Joint Precision score: {edge_joint_precision:.4f}")
    print(f"Edge Joint Recall score: {edge_joint_recall:.4f}")
    print(f"Entity F1 score: {f1:.4f}")
    #print(f"Edge F1 score: {edge_f1:.4f}")
    print(f"Edge Joint F1 score: {edge_joint_f1:.4f}")

    
    #print(classification_report(all_labels, all_preds))
    model.train()
    return f1, edge_joint_f1



