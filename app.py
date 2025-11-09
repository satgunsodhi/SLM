import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import streamlit as st
import tiktoken
from safetensors.torch import load_file
import huggingface_hub


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Flash attention support (much faster)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # Causal mask for non-flash attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate Q, K, V for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Efficient attention
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                dropout_p=self.attn_dropout.p if self.training else 0.0, 
                                                is_causal=True)
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block: attention + MLP with residual connections"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    """GPT model configuration"""
    block_size: int       # Max sequence length
    vocab_size: int       # Vocabulary size
    n_layer: int          # Number of transformer blocks
    n_head: int           # Number of attention heads
    n_embd: int           # Embedding dimension
    dropout: float = 0.0  # Dropout rate
    bias: bool = True     # Use bias in linear layers


class GPT(nn.Module):
    """GPT Language Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share embeddings with output layer
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward pass
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference mode: only compute last token
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
                        
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============= Streamlit App =============

# Page config
st.set_page_config(
    page_title="FinPeak.ai",
    page_icon="📈",
    layout="wide"
)


# Cache the model loading
@st.cache_resource
def load_model():
    """Download and load model from Hugging Face using SafeTensors"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Download SafeTensors model from Hugging Face Hub
    with st.spinner("Downloading model from Hugging Face... (first time only)"):
        model_path = huggingface_hub.hf_hub_download(
            repo_id="satgun/finpeak-ai",
            filename="model.safetensors",  # Changed from .pt to .safetensors
            cache_dir="./model_cache"
        )
        
        # Also download config if available (optional but recommended)
        try:
            config_path = huggingface_hub.hf_hub_download(
                repo_id="satgun/finpeak-ai",
                filename="config.json",
                cache_dir="./model_cache"
            )
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
                config = GPTConfig(**config_dict)
        except:
            # Fallback to default config if config.json not available
            config = GPTConfig(
                block_size=256,
                vocab_size=50257,
                n_layer=8,
                n_head=12,
                n_embd=768,
                dropout=0.14,
                bias=True
            )
    
    # Load weights from SafeTensors
    state_dict = load_file(model_path, device=device)
    
    # Create and load model
    model = GPT(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    return model, enc, device


# Initialize model
model, enc, device = load_model()


# ============= Header with Branding =============

st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='margin: 0; font-size: 3rem;'>📈 FinPeak<span style='color: #666;'>.ai</span></h1>
    <p style='color: #888; margin-top: 0.5rem;'>Your AI-Powered Financial Intelligence Assistant</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Generation function
def generate_response(prompt, max_tokens=200, temperature=0.7):
    """Generate response from the model"""
    formatted_prompt = f"""### Instruction:
{prompt}

### Input:

### Response:
"""
    
    context = torch.tensor(enc.encode_ordinary(formatted_prompt), 
                          dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_tokens, 
                               temperature=temperature)
    
    full_response = enc.decode(output[0].tolist())
    
    # Extract response
    if "### Response:" in full_response:
        response = full_response.split("### Response:")[-1]
        # Stop at "### End" or next instruction
        for stop_word in ["### End", "### Instruction", "\n\n\n"]:
            if stop_word in response:
                response = response.split(stop_word)[0]
        return response.strip()
    
    return "I couldn't generate a proper response. Please try rephrasing your question."


# Chat input
if prompt := st.chat_input("Ask me anything about finance, investing, or company analysis..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = generate_response(
                prompt,
                max_tokens=st.session_state.get('max_tokens', 200),
                temperature=st.session_state.get('temperature', 0.7)
            )
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})


# Sidebar with settings
with st.sidebar:
    st.markdown("### ⚙️ Generation Settings")
    
    st.session_state.temperature = st.slider(
        "Temperature", 0.1, 1.0, 0.7, 0.1,
        help="Higher = more creative, Lower = more focused"
    )
    st.session_state.max_tokens = st.slider(
        "Max Response Length", 50, 500, 200, 50,
        help="Maximum tokens to generate"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Device:** `{device}`")
    st.markdown(f"**Vocab Size:** `{enc.n_vocab:,}`")
    st.markdown(f"**Format:** `SafeTensors`")
    st.markdown(f"**Model:** Custom GPT")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Example prompts in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Try Asking")

example_prompts = [
    "What is a P/E ratio?",
    "Should I invest in tech stocks?",
    "Explain portfolio diversification",
    "What are red flags in financial statements?",
    "How to analyze a company's balance sheet?",
    "Difference between stocks and bonds?"
]

for example in example_prompts:
    if st.sidebar.button(example, key=example, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": example})
        response = generate_response(
            example,
            max_tokens=st.session_state.get('max_tokens', 200),
            temperature=st.session_state.get('temperature', 0.7)
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    <p>Powered by <strong>FinPeak.ai</strong></p>
    <p style='font-size: 0.75rem;'>Custom GPT Model • SafeTensors</p>
</div>
""", unsafe_allow_html=True)
