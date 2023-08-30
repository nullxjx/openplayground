import numpy as np
from transformers import AutoTokenizer
from .common import prepare_tensor
from .common import create_inference_server_client


tokenizer = AutoTokenizer.from_pretrained("./tokenizers/llama")


def prepare_final_inputs(prompt, protocol, batch_size=1, start_id=1, end_id=2, topK=1, topP=0, return_log_probs=False,
                         beamWidth=1):
    tasks = []
    for i in range(batch_size):
        task = {'tokens': tokenizer.encode(prompt)}
        tasks.append(task)

    input_lens = [len(task['tokens']) for task in tasks[:batch_size]]
    max_len = max(input_lens)
    input_tokens = []
    for task in tasks[:batch_size]:
        input_tokens.append(task['tokens'] + [0] * (max_len - len(task['tokens'])))
    request_lens = [512] * batch_size

    input_ids = np.array(input_tokens, dtype=np.uint32)
    input_lens = np.array(input_lens, dtype=np.uint32).reshape(-1, 1)
    request_lens = np.array(request_lens, dtype=np.uint32).reshape(-1, 1)
    runtime_top_k = (topK * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_top_p = topP * np.ones([batch_size, 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([batch_size, 1]).astype(np.float32)
    temperature = 0 * np.ones([batch_size, 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([batch_size, 1]).astype(np.float32)
    random_seed = 0 * np.ones([batch_size, 1]).astype(np.uint64)
    is_return_log_probs = return_log_probs * np.ones([batch_size, 1]).astype(bool)
    beam_width = (beamWidth * np.ones([batch_size, 1])).astype(np.uint32)
    start_ids = start_id * np.ones([batch_size, 1]).astype(np.uint32)
    end_ids = end_id * np.ones([batch_size, 1]).astype(np.uint32)
    bad_words_ids = np.array([[[0], [-1]]] * batch_size, dtype=np.int32)
    stop_words_ids = np.array([[[0], [-1]]] * batch_size, dtype=np.int32)

    return [
        prepare_tensor("input_ids", input_ids, protocol),
        prepare_tensor("input_lengths", input_lens, protocol),
        prepare_tensor("request_output_len", request_lens, protocol),
        prepare_tensor("runtime_top_k", runtime_top_k, protocol),
        prepare_tensor("runtime_top_p", runtime_top_p, protocol),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, protocol),
        prepare_tensor("temperature", temperature, protocol),
        prepare_tensor("len_penalty", len_penalty, protocol),
        prepare_tensor("repetition_penalty", repetition_penalty, protocol),
        prepare_tensor("random_seed", random_seed, protocol),
        prepare_tensor("is_return_log_probs", is_return_log_probs, protocol),
        prepare_tensor("beam_width", beam_width, protocol),
        prepare_tensor("start_id", start_ids, protocol),
        prepare_tensor("end_id", end_ids, protocol),
        prepare_tensor("bad_words_list", bad_words_ids, protocol),
        prepare_tensor("stop_words_list", stop_words_ids, protocol),
    ]


def infer(infer_url, prompt, model, topK=1, topP=0, return_log_probs=False, beamWidth=1):
    protocol = "http"
    with create_inference_server_client(protocol, infer_url, concurrency=1, verbose=False) as client:
        try:
            inputs = prepare_final_inputs(prompt, protocol)
            result = client.infer(model, inputs)
        except Exception as e:
            print(e)
