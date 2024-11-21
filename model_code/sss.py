import json
import esm
import itertools
import torch
from esm import ESM2
foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
def load_foldseek_ckpt(ckpt_path, config_path):
    config = json.load(open(config_path, "r"))
    
    # Initialize the alphabet
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
    for seq_token, struc_token in itertools.product(foldseek_seq_vocab, foldseek_struc_vocab):
        token = seq_token + struc_token
        tokens.append(token)
    
    alphabet = esm.data.Alphabet(standard_toks=tokens,
                                prepend_toks=[],
                                append_toks=[],
                                prepend_bos=True,
                                append_eos=True,
                                use_msa=False)
    
    alphabet.all_toks = alphabet.all_toks[:-2]
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}
    
    # Initialize the model
    model = ESM2(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        alphabet=alphabet,
        token_dropout=config["token_dropout"],
    )
    
    # Load the checkpoint
    model_data = torch.load(ckpt_path, map_location="cpu")
    
    # Convert the keys
    weights = {k.replace("esm.encoder.layer", "layers"): v for k, v in model_data["model"].items()}
    weights = {k.replace("attention.self", "self_attn"): v for k, v in weights.items()}
    weights = {k.replace("key", "k_proj"): v for k, v in weights.items()}
    weights = {k.replace("query", "q_proj"): v for k, v in weights.items()}
    weights = {k.replace("value", "v_proj"): v for k, v in weights.items()}
    weights = {k.replace("attention.output.dense", "self_attn.out_proj"): v for k, v in weights.items()}
    weights = {k.replace("attention.LayerNorm", "self_attn_layer_norm"): v for k, v in weights.items()}
    weights = {k.replace("intermediate.dense", "fc1"): v for k, v in weights.items()}
    weights = {k.replace("output.dense", "fc2"): v for k, v in weights.items()}
    weights = {k.replace("LayerNorm", "final_layer_norm"): v for k, v in weights.items()}
    weights = {k.replace("esm.embeddings.word_embeddings", "embed_tokens"): v for k, v in weights.items()}
    weights = {k.replace("rotary_embeddings", "rot_emb"): v for k, v in weights.items()}
    weights = {k.replace("embeddings.LayerNorm", "embed_layer_norm"): v for k, v in weights.items()}
    weights = {k.replace("esm.encoder.", ""): v for k, v in weights.items()}
    weights = {k.replace("lm_head.decoder.weight", "lm_head.weight"): v for k, v in weights.items()}
    for k in ["esm.embeddings.position_ids", "esm.embeddings.position_embeddings.weight"]:
        if k in weights:
            weights.pop(k)
    
    model.load_state_dict(weights, strict=False)
    return model, alphabet