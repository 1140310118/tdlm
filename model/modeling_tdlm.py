import math
import torch
from torch import nn
from einops import rearrange, repeat

from performer_pytorch import FastAttention
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.deberta.modeling_deberta import XSoftmax
from transformers.models.deberta.modeling_deberta import StableDropout



class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]

        elif x.dim() == 4:
            # return x + self.embeddings(position_ids)[None, None, :, :]
            h = x.size(1)
            x = rearrange(x, 'b h l d -> b l (h d)')
            x = x + self.embeddings(position_ids)[None, :, :]
            x = rearrange(x, 'b l (h d) -> b h l d', h=h)
            return x



class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, position_embedding_type):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def forward_rope(self, x):
        """
        return (b l d)
        """
        embeddings = self.embeddings[None, :x.size(1), :] # b l d
        embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
        sin, cos = embeddings.unbind(dim=-2) # b l d//2
        sin, cos = map(lambda t: repeat(t, '... d -> ... (d 2)'), (sin, cos)) # b l d
        return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x):
        if self.position_embedding_type == 'fixed':
            return self.forward_fixed(x)

        elif self.position_embedding_type == 'rope':
            return self.forward_rope(x)

    def forward(self, x):
        if x.dim() == 3:
            return self._forward(x)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x



class RelativePositionEmbedding(nn.Module):
    def __init__(self, 
                 relative_attention_num_buckets, num_attention_heads, 
                 hidden_size, position_embedding_type):

        super().__init__()

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.position_embedding_type = position_embedding_type
        self.num_attention_heads = num_attention_heads
        self.is_absolute = False

        if position_embedding_type == 'bias':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, num_attention_heads)

        elif position_embedding_type == 'contextual(1)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)
            self.to_r = nn.Linear(hidden_size, hidden_size, bias=False)

        elif position_embedding_type == 'contextual(2)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)
            # self.to_kr = nn.Linear(hidden_size, hidden_size, bias=False)
            # self.to_qr = nn.Linear(hidden_size, hidden_size, bias=False)

            # self.to_kr.weight = to_k.weight
            # self.to_qr.weight = to_q.weight

    def compute_bias(self, q, k, to_q=None, to_k=None):
        """
        q, k: [b h l d]
        return [b h l l]
        """
        h = self.num_attention_heads
        query_position = torch.arange(q.size(2), dtype=torch.long, device=self.embeddings.weight.device)[:, None]
        key_position   = torch.arange(k.size(2), dtype=torch.long, device=self.embeddings.weight.device)[None, :]

        relative_position = query_position - key_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets
        )

        if self.position_embedding_type == 'bias':
            bias = self.embeddings(relative_position_bucket)
            bias = rearrange(bias, 'm n h -> 1 h m n')

        elif self.position_embedding_type == 'contextual(1)':
            r = self.embeddings(relative_position_bucket)
            r = self.to_r(r)
            r = rearrange(r, 'm n (h d) -> h m n d', h=h)

            bias = torch.einsum('b h m d, h m n d -> b h m n', q, r)

        elif self.position_embedding_type == 'contextual(2)':
            r = self.embeddings(relative_position_bucket)

            kr = to_k(r)
            qr = to_q(r)

            kr = rearrange(kr, 'm n (h d) -> h m n d', h=h)
            qr = rearrange(qr, 'm n (h d) -> h m n d', h=h)

            bias1 = torch.einsum('b h m d, h m n d -> b h m n', q, kr)
            bias2 = torch.einsum('b h n d, h m n d -> b h m n', k, qr)

            bias = bias1 + bias2

        return bias

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance=128):
        """
        relative_position: [m n]
        """

        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets




class TDLMConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        hidden_size=512,
        num_hidden_layers=8,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        glu=False,
        position_embedding_type='learnable',
        encoder_layer='transformer',
        pre_norm=True,
        relative_attention_num_buckets=32,
        **kwargs
    ):

        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.glu = glu
        
        assert position_embedding_type in (
            'learnable', 'fixed', 'rope',
            'layerwise_learnable', 'layerwise_fixed', 'layerwise_rope',
            'layerwise_bias', 'layerwise_contextual(1)', 'layerwise_contextual(2)',
        )
        self.position_embedding_type = position_embedding_type
        self.encoder_layer = encoder_layer
        self.pre_norm = pre_norm
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.model_type = 'tdlm'




class Embedddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.embedding_size, config.hidden_size)
        
        if config.position_embedding_type == 'learnable':
            self.position_embeddings = LearnableAbsolutePositionEmbedding(
                max_position_embeddings=config.max_position_embeddings, 
                hidden_size=config.hidden_size
            )
        
        elif config.position_embedding_type in ('fixed', 'rope'):
            self.position_embeddings = FixedAbsolutePositionEmbedding(
                max_position_embeddings=config.max_position_embeddings,
                hidden_size=config.hidden_size,
                position_embedding_type=config.position_embedding_type
            )

    def forward(self, input_ids):
        embeds = self.word_embeddings(input_ids)
        embeds = self.dropout(embeds)
        embeds = self.dense(embeds)

        if hasattr(self, 'position_embeddings'):
            embeds = self.position_embeddings(embeds)

        return embeds






class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_heads = config.num_attention_heads
        dim_heads = config.hidden_size // config.num_attention_heads

        self.to_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.to_out  = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = StableDropout(config.hidden_dropout_prob)

        # if config.position_embedding_type == 'layerwise_learnable':
        #     self.position_embeddings = LearnableAbsolutePositionEmbedding(
        #         max_position_embeddings=config.max_position_embeddings, 
        #         hidden_size=dim_heads
        #     )
        
        # elif config.position_embedding_type in ('layerwise_fixed', 'layerwise_rope'):

        #     self.position_embeddings = FixedAbsolutePositionEmbedding(
        #         max_position_embeddings=config.max_position_embeddings,
        #         hidden_size=dim_heads,
        #         position_embedding_type=config.position_embedding_type.split('_')[-1],
        #     )

        # elif config.position_embedding_type in ('layerwise_bias', 'layerwise_contextual(1)', 'layerwise_contextual(2)'):
        #     self.position_embeddings = RelativePositionEmbedding( 
        #          config.relative_attention_num_buckets, 
        #          config.num_attention_heads, 
        #          config.hidden_size, 
        #          position_embedding_type=config.position_embedding_type.split('_')[-1],
        #          to_q=self.to_q, 
        #          to_k=self.to_k
        #     )

        if config.encoder_layer == 'transformer':
            self.attn_fn = TransformerAttention(config)

        elif config.encoder_layer == 'performer':
            self.attn_fn = PerformerAttention(config)

        else:
            raise NotImplementedError


    def forward(self, x, mask, pos_emb):
        h = self.n_heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=h), (q, k, v))

        context = self.attn_fn(q, k, v, mask, pos_emb, to_q=self.to_q, to_k=self.to_k)
        out = self.to_out(context)
        out = self.dropout(out)

        return out




class TransformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        attention_head_size = config.hidden_size // config.num_attention_heads
        self.scale = attention_head_size ** -0.5
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def forward(self, q, k, v, mask, pos_emb, to_q, to_k):
        """
        q, k, v: [b h l d]
        mask: [b l]
        """
        if pos_emb is not None and pos_emb.is_absolute is True:
            q = pos_emb(q)
            k = pos_emb(k)

        dots = torch.einsum('b h m d, b h n d -> b h m n', q, k)

        if pos_emb is not None and pos_emb.is_absolute is False:
            bias = pos_emb.compute_bias(q, k, to_q, to_k)
            dots = dots + bias

        # assert mask is not None
        # if mask is not None:
        mask = mask[:, None, None, :] & mask[:, None, :, None]
        # dots = dots.masked_fill(~mask, -10000.)
        # probs = dots.softmax(dim=-1)
        probs = XSoftmax.apply(dots, mask, -1)

        probs = self.dropout(probs)

        context = torch.einsum('b h m n, b h n d -> b h m d', probs, v)
        context = rearrange(context, 'b h m d -> b m (h d)')

        return context



class PerformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        attention_head_size = config.hidden_size // config.num_attention_heads
        self.attn = FastAttention(dim_heads=attention_head_size, causal=False)

    def forward(self, q, k, v, mask, pos_emb, **kwargs):
        """
        q, k, v: [b h l d]
        mask: [b l]
        """
        if pos_emb is not None:
            assert pos_emb.is_absolute is True
            q = pos_emb(q)
            k = pos_emb(k)

        mask = mask[:, None, :, None]
        v = v.masked_fill(~mask, 0.)

        context = self.attn(q, k, v)
        context = rearrange(context, 'b h l d -> b l (h d)')
        return context



class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.glu = config.glu

        if config.glu:
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size*8)
        else:
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size*4)

        self.dense2 = nn.Linear(config.hidden_size*4, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = StableDropout(config.hidden_dropout_prob)

        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.glu:
            x, v = self.dense1(x).chunk(2, dim=-1)
            x = self.intermediate_act_fn(x) * v

        else:
            x = self.dense1(x)
            x = self.intermediate_act_fn(x)

        x = self.dense2(x)
        x = self.dropout(x)

        return x



class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_norm = config.pre_norm

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attention = Attention(config)
        self.feedforward = FeedForward(config)

    def forward(self, x, mask, pos_emb):
        if self.pre_norm is False:
            x = self.norm1(self.attention(x, mask, pos_emb) + x)
            x = self.norm2(self.feedforward(x) + x)

        else:
            x = self.attention(self.norm1(x), mask, pos_emb) + x
            x = self.feedforward(self.norm2(x)) + x

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim_heads = config.hidden_size // config.num_attention_heads 

        if config.position_embedding_type == 'layerwise_learnable':
            self.position_embeddings = LearnableAbsolutePositionEmbedding(
                max_position_embeddings=config.max_position_embeddings, 
                # hidden_size=dim_heads
                hidden_size=config.hidden_size
            )
        
        elif config.position_embedding_type in ('layerwise_fixed', 'layerwise_rope'):
            self.position_embeddings = FixedAbsolutePositionEmbedding(
                max_position_embeddings=config.max_position_embeddings,
                hidden_size=dim_heads,
                position_embedding_type=config.position_embedding_type.split('_')[-1],
            )

        elif config.position_embedding_type in ('layerwise_bias', 'layerwise_contextual(1)', 'layerwise_contextual(2)'):
            self.position_embeddings = RelativePositionEmbedding( 
                 config.relative_attention_num_buckets, 
                 config.num_attention_heads, 
                 config.hidden_size, 
                 position_embedding_type=config.position_embedding_type.split('_')[-1],
                 # to_q=self.to_q,
                 # to_k=self.to_k
            )

        else:
            self.position_embeddings = None

        self.layer = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, mask):
        for layer_module in self.layer:
            x = layer_module(x, mask, self.position_embeddings)

        return x



class TDLMPreTrainedModel(PreTrainedModel):
    config_class = TDLMConfig
    base_model_prefix = 'tdlm'

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, FixedAbsolutePositionEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



class TDLMModel(TDLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = Embedddings(config)
        self.encoder  = Encoder(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [b, l]
        attention_mask: [b, l]
        """
        attention_mask = attention_mask.bool()

        x = self.embeddings(input_ids)
        x = self.encoder(x, attention_mask)

        return x



class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform_dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform_dense(x)
        x = self.transform_act_fn(x)
        x = self.norm(x)

        scores = self.decoder(x)
        return scores


class TDLMForPretraining(TDLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.tdlm = TDLMModel(config)
        self.mlm_head = MLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self.mlm_head.decoder.weight = self.tdlm.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        x = self.tdlm(input_ids, attention_mask)
        scores = self.mlm_head(x)

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return loss


