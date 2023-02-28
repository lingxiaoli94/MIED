import torch
from tqdm import tqdm

def batch_eval_index(f, total_count, batch_size=1024,
                     result_device=torch.device('cpu'),
                     detach=True,
                     no_tqdm=False):
    '''
    Batch evaluate f.

    :param f: function to be evalutated. It should take in (B,) of indices.
    :param total_count: total number of indices
    :param batch_size: batch size in each invocation of f
    :return: a list of results. You might want to call torch.cat afterwards.
    '''
    result = []
    current_count = 0
    with tqdm(total=total_count, disable=no_tqdm) as pbar:
        while current_count < total_count:
            count = min(batch_size, total_count - current_count)
            inds = slice(current_count, current_count + count)
            cur_result = f(inds)
            if detach:
                cur_result = cur_result.detach()
            result.append(cur_result.to(result_device))
            current_count += count
            pbar.update(count)
        pbar.close()

    return result
