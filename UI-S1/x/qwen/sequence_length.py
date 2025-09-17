from .tokenizer import get_text_seq_len

def get_sequence_length_rich(ele):
    if 'image' in ele:
        if 'seq_len' not in ele:
            from .image import update_image_size_
            ele = update_image_size_(ele)
        return ele['seq_len']
    elif ('text' in ele) or ('prompt' in ele):
        return get_text_seq_len(ele)
