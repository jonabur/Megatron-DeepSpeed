import sys
from pathlib import Path
import torch
from collections import OrderedDict
import re
import argparse
import os
## Assumptions: 
# swiglu
# rotary positional
# embeddings
    # Search in directory above this
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                    os.path.pardir)))

PARALLEL_RANK_PATTERN = re.compile("mp_rank_\d*_\d*")
CP_ID_PATTERN = re.compile("iter_\d*")

DEVICE = 'cpu'

def recursive_print(m, level):
    if type(m) == dict or type(m) == OrderedDict:
        for k, v in m.items():
            if level==0:
                if 'model' in k:
                    print(f'{"#", "#"*(level*4), k}', flush=True)
                    recursive_print(v, level=level+1)
            else:
                if type(v) == torch.Tensor:
                    print(f'{"#"*(level*4), k, v.shape}')
                else:
                    print(f'{"#"*(level*4), k}')
                recursive_print(v, level=level+1)

def parse_output_path(args):
    iter_id = CP_ID_PATTERN.search(args.path_to_checkpoint).group()
    out = args.output_path
    output_path = os.path.join(out, iter_id)
    return output_path

def add_or_combine_to_dict(target, shard, target_key, dim=0):
    target_value = target.get(target_key)
    # key = new_key if new_key else target_key
    if target_value != None:
        target[target_key] = torch.cat([target_value, shard], dim=dim)
    else:
        target[target_key] = shard

def combine_swiglu_mlp(encoder):
    up_layer_keys = sorted([k for k in encoder.keys() if "h_to_4h.weight.up_proj" in k])
    gate_layer_keys = sorted([k for k in encoder.keys() if "h_to_4h.weight.gate_proj" in k])

    for (up_key, gate_key) in zip(up_layer_keys, gate_layer_keys):
        up = encoder.pop(up_key)
        gate = encoder.pop(gate_key)
        # delete temp proj keys
        encoder[".".join(up_key.split(".")[:-1])] = torch.cat([up, gate], dim=0)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "output_path",
        help='Path to the output directory to store the converted checkpoint'
    )


    args = parser.parse_args()
    chunks = [pt.absolute() for pt in Path(args.path_to_checkpoint).glob("*/*")]
    chunks = sorted(chunks)
    cp_args = None 
    vp_size = 0
    tp_size = 0
    pp_size = 0
    vp_layers = 0
    encoder = {}
    word_embeddings = {}
    output_layer = {}
    output_path = parse_output_path(args)
    iteration = ""
    cp_version = ""
    tokens = ""
    # Looping order: TP_PP 00_000, 00_001, ..., 01_000, ... 
    for i, chunk in enumerate(chunks):
        tp_rank, pp_rank = PARALLEL_RANK_PATTERN.search(str(chunk)).group().split("_")[-2:]
        tp_rank, pp_rank = int(tp_rank), int(pp_rank)

        print("processing #", chunk.absolute())
        print(f"tp_rank: {tp_rank}, pp_rank: {pp_rank}")

        shard = torch.load(chunk, map_location='cuda')
        if i == 0:
            cp_args = shard['args']
            vp_size = cp_args.virtual_pipeline_model_parallel_size
            vp_layers = cp_args.num_layers_per_virtual_pipeline_stage
            pp_size = cp_args.pipeline_model_parallel_size
            tp_size = cp_args.tensor_model_parallel_size
            num_layers = cp_args.num_layers
            iteration = shard['iteration']
            cp_version = shard['checkpoint_version']
            tokens = shard['tokens']
        
        if vp_size == 1:
            #TODO
            assert False, "Not yet implemented"
            lm = shard['model']['language_model']
        
        else:
            for vp_rank in range(vp_size):
                # test vp offsetting by having model splitted up into layers / (pp_size *vp_layers). With layers=40, pp=4, vp_layers=5 -> model is split into shards: 5x4x2
                vp_offset = vp_rank * (pp_size * vp_layers) 
                pp_offset = pp_rank * vp_layers
                conv = lambda s: f".{str(int(s.groups()[0]) + pp_offset + vp_offset)}."
                lm = shard[f'model{vp_rank}']['language_model']
                print(f"model{vp_rank}, {pp_rank}, {vp_rank}")
                if vp_rank == 0 and pp_rank == 0:
                    # handle word embeddings / tensor parallel level
                    embedding_shard = lm['embedding']['word_embeddings']['weight']
                    add_or_combine_to_dict(word_embeddings, embedding_shard, target_key='weight')

                    # stored_embedding = word_embeddings.get('weight')
                    # if stored_embedding != None:
                    #     word_embeddings = {"weight" : torch.cat([stored_embedding, embedding_shard], dim=0)}
                    # else:
                    #     word_embeddings = {"weight": embedding_shard}

                if pp_rank == (pp_size-1) and vp_rank == (vp_size-1):
                    # convert Namespace-object to dict 
                    if vars(cp_args).get('untie_embeddings_and_output_weights'):
                        print("Having untied embeddings")
                        output_layer_shard = lm['output_layer']['weight']
                        add_or_combine_to_dict(output_layer, output_layer_shard, target_key='weight')
                    # stored_output_layer = output_layer.get('weight')
                    # if stored_output_layer != None:
                    #     output_layer = {'weight': torch.cat([stored_output_layer, output_layer_shard], dim=0)}
                    # else:
                    #     output_layer = {"weight": output_layer_shard}

                
                for name, layer in lm['encoder'].items():
                    
                    layer = layer.to(DEVICE)
                    layer_name = re.sub("\.(\d*)\.", conv, name)
                    # state_dict_layer = encoder.get(layer_name)
    

                    if cp_args.swiglu:
                        if "mlp.dense_h_to_4h" in name:
                            up_proj, gate_proj = torch.chunk(layer, 2, dim=0)
                            print("MLP shapes:", up_proj.shape, gate_proj.shape)
                            up_proj_key = layer_name + ".up_proj"
                            gate_proj_key = layer_name + ".gate_proj"
                            add_or_combine_to_dict(encoder, up_proj, up_proj_key, dim=0)
                            add_or_combine_to_dict(encoder, gate_proj, gate_proj_key, dim=0)
                        else:
                            if ('self_attention.dense.weight' in name) or \
                                ('mlp.dense_4h_to_h' in name):
                                add_or_combine_to_dict(encoder, layer, layer_name, dim=1)
                            elif "layernorm" in layer_name:
                                if tp_rank == 0:
                                    # only take layernorms from the first layers
                                    add_or_combine_to_dict(encoder, layer, layer_name)
                            else:
                                add_or_combine_to_dict(encoder, layer, layer_name, dim=0)
   
                    else:
                        # TODO: swiglu is a speacial case -> generalize
                        add_or_combine_to_dict(encoder, layer, layer_name)

                        
    # encoder['output_layer'] = output_layer 
    cp_args.pipeline_model_parallel_size = 1
    cp_args.tensor_model_parallel_size = 1
    combine_swiglu_mlp(encoder)
    # Combine into a single state_dict
    state_dict = { 
        "model": {
            "language_model": {
                "embedding" : {
                    "word_embeddings": word_embeddings
                },
                "encoder": encoder,
                "output_layer": output_layer

            }
        },
        "args": cp_args,
        "iteration": iteration,
        "checkpoint_version": cp_version,
        "tokens": tokens
    }
    
    if not os.path.exists(os.path.join(output_path, "mp_rank_00")):
        os.makedirs(os.path.join(output_path, "mp_rank_00"))
    
    # Save latest iteration for megatron loader
    iter_path =os.path.join('/'.join(output_path.split("/")[:-1]), 'latest_checkpointed_iteration.txt')
    with open(iter_path, 'w') as c_out:
        c_out.write(str(iteration))
    parsed_output_path = os.path.join(output_path, "mp_rank_00", "model_optim_rng.pt") 
    torch.save(state_dict, parsed_output_path)
    print(f"Succesfully saved the model to {parse_output_path}")

                






        

if __name__ == '__main__':
    main()