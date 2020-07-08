import torch
import torch.nn as nn


class embedding_distill(nn.Module):
    def __init__(self, args):
        super (embedding_distill, self).__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(30522, self.args.ext_hidden_size)
        self.position_embeddings = nn.Embedding(self.args.max_pos, self.args.ext_hidden_size)
        self.token_type_embeddings = nn.Embedding(30522, self.args.ext_hidden_size)

        self.LayerNorm = nn.LayerNorm(self.args.ext_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, segs, position_ids=None):
        input_shape = x.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embeddings(x)

        device = x.device if x is not None else inputs_embeds.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings = self.position_embeddings(position_ids)
        
        token_type_embeddings = self.token_type_embeddings(segs)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
