from transformers import BertModel, AutoConfig
import torch
import torch.nn as nn
import train
import sgen_model

def train_model(train_loader, val_loader, test_loader, universal_pos_tags, iob_labels, iob2id, id2iob, relation_type_mapping, depsize):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model, loss, and optimizer
    input_dim = 768 + len(universal_pos_tags) #768 + 5 BERT // 300 + 5 spacy
    hidden_dim = 768 + len(universal_pos_tags)#  512 spacy // 1024 BERT
    model_checkpoint = "neuralmind/bert-base-portuguese-cased"  # or any other model checkpoint
    config = AutoConfig.from_pretrained(model_checkpoint) # lfcc/lusa_events neuralmind/bert-base-portuguese-cased
    config.num_labels = len(iob_labels)  # Set the number of labels for token classification
    config.id2label = id2iob
    config.label2id = iob2id
    bert = BertModel.from_pretrained(
        model_checkpoint,
        config=config,
        add_pooling_layer=True  # Required to match pretrained checkpoint
    )
    model = sgen_model.SGEN(input_dim, hidden_dim, config, dep_size= depsize,
                            relation_type_mapping=relation_type_mapping,
                            iob_labels= iob_labels
                        )
    missing_keys, unexpected_keys = model.bert.load_state_dict(
        bert.state_dict(), strict=False
    )
    model = model.to(device)


    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("bert.")], "lr": 1e-3}
    ])
    
    label_loss_fn = nn.BCEWithLogitsLoss()
    entity_loss_fn = nn.CrossEntropyLoss()

    model_save_path = "./models/__.pt"
    best_val_f1 = 0.0
    patience = 20   # stop if no improvement for 30 epochs
    counter = 0

    for epoch in range(1000):
        # Training step
        loss = train.train(model, train_loader, label_loss_fn, entity_loss_fn, optimizer, scheduler=None)
        
        if epoch % 1 == 0:
            print(f"=================== EPOCH {epoch} ==================")
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Evaluate on validation set
            val_entity_f1, val_edge_f1 = train.evaluate(model, val_loader, device)
            #print(f"Validation Edge F1: {val_edge_f1:.4f}, Entity F1: {val_entity_f1:.4f}")

            # Early stopping / best model saving
            if val_edge_f1 > best_val_f1:
                best_val_f1 = val_edge_f1
                torch.save(model.state_dict(), model_save_path)
                print(f"Best model saved with Validation Edge F1: {best_val_f1:.4f}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break
            #if epoch % 10 == 0:
            #    # Evaluate on test set every 10 epochs
            #    print(" ==================== Evaluating on test set... ==================")
            #    test_entity_f1, test_edge_f1 = evaluate_bert(model, test_loader, device, use_entity_preds=True)
            #    print(f"Test Edge F1: {test_edge_f1:.4f}, Entity F1: {test_entity_f1:.4f}")
            #    print(" =================================================================")

    # Load the best model before final test evaluation
    model.load_state_dict(torch.load(model_save_path))
    test_entity_f1, test_edge_f1 = train.evaluate(model, test_loader, device, pipeline=False)
    print(f"Final Test Edge F1: {test_edge_f1:.4f}, Entity F1: {test_entity_f1:.4f}")
