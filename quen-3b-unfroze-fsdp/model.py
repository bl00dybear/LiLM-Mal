import torch
import torch.nn as nn
from transformers import Qwen2Model 

class MalwareDetectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.model = Qwen2Model.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

        self.model.config.use_cache = False

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        for param in self.model.parameters():
            param.requires_grad = False

        layers = self.model.layers
        total_layers = len(layers)
        start_unfreeze = max(0, total_layers - config.n_unfrozen_layers)

        for i in range(start_unfreeze, total_layers):
            for param in layers[i].parameters():
                param.requires_grad = True

        hidden_dim = self.model.config.hidden_size
        
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=torch.bfloat16),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, dtype=torch.bfloat16)
        )

        self.regression_head = nn.Linear(hidden_dim, 1, dtype=torch.bfloat16)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)

        nn.init.xavier_uniform_(self.attention_net[0].weight)
        nn.init.xavier_uniform_(self.attention_net[2].weight)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size, num_chunks, seq_len = input_ids.shape
        
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)

        outputs = self.model(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
            return_dict=True
        )

        last_hidden = outputs.last_hidden_state 
        
        chunk_embeddings = last_hidden[:, -1, :]
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)

        attn_weights = self.attention_net(chunk_embeddings)
        attn_weights = torch.softmax(attn_weights, dim=1)

        pooled_output = torch.sum(attn_weights * chunk_embeddings, dim=1)
        logits = self.regression_head(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.to(logits.dtype))

        return {"loss": loss, "logits": logits} if loss is not None else logits