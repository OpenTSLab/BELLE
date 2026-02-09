import torch
import os


def resize_position_weight(model_data, weight_name, new_length, verbose=True):
    """
    Resize positional weight shape.

    Args:
        model_data: Model data dictionary.
        weight_name: Weight name (e.g., 'text_position.weight').
        new_length: New length.
        verbose: Whether to print details.

    Returns:
        Resized weight matrix.
    """
    original_weight = model_data['model'][weight_name]
    original_shape = original_weight.shape
    
    if verbose:
        print(f"\nProcessing {weight_name}:")
    
    if new_length > original_shape[0]:
        # If new length is longer, create a new weight matrix
        new_weight = torch.randn(new_length, original_shape[1])  # Randomly initialize new weights
        # Load original weights on the left
        new_weight[:original_shape[0], :] = original_weight
        
        if verbose:
            print(f"Resized weight shape: {original_shape} -> {new_weight.shape}")
            print(f"Original weights loaded to [0:{original_shape[0]}, :]")
            print(f"New weights randomly initialized at [{original_shape[0]}:{new_length}, :]")
    else:
        # If new length is shorter or equal, truncate directly
        new_weight = original_weight[:new_length, :]
        if verbose:
            print(f"Truncated weight shape: {original_shape} -> {new_weight.shape}")
    
    # Update model weights
    model_data['model'][weight_name] = new_weight
    
    if verbose:
        print(f"Final {weight_name} weight shape: {new_weight.shape}")
    
    return new_weight


# Main program
input_file = "egs/librispeech/exp/belle-lr5e-4-kl0-edl0.2-flux0.5-epoch60-cuts_train_filter_all-tts_models_cosyvoice_indextts_sparktts_f5tts_xtts_maskgct-loss_weight_0.22_0.13_0.13_0.13_0.13_0.13_0.13-coef0.5-inversegamma/epoch-60.pt"
output_file = "egs/librispeech/exp/belle_stream-lr5e-4-flux0.5-epoch10-cuts_train_filter_all_with_prompts-tts_models_cosyvoice_indextts_sparktts_f5tts_xtts_maskgct-loss_weight_0.22_0.13_0.13_0.13_0.13_0.13_0.13-train_stage2-nframes1-textchunk20-audiochunk50/epoch-1.pt"

input_data = torch.load(input_file, map_location="cpu", weights_only=False)
print(input_data.keys())
print("Original weight shapes:")
print(f"text_position.weight: {input_data['model']['text_position.weight'].shape}")
print(f"audio_position.weight: {input_data['model']['audio_position.weight'].shape}")

text_embedding = 2000
audio_embedding = 5000

# Resize weight shapes
resize_position_weight(input_data, 'text_position.weight', text_embedding)
resize_position_weight(input_data, 'audio_position.weight', audio_embedding)
input_data['text_embeddings'] = text_embedding
input_data['audio_embeddings'] = audio_embedding
input_data['text_tokens'] = 'data/tokenized/unique_text_tokens_libriheavy.k2symbols'

# Save updated model
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(input_data, output_file)
print(f"\nUpdated model saved to: {output_file}")
