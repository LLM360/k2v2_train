import json
import random
import os
from optparse import OptionParser
import sys
import time
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("LLM360/K2-V2-Instruct")
SYSTEM = 'You are K2, a helpful assistant created by Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) Institute of Foundation Models (IFM).'

def parse_args():
    parser = OptionParser()
    
    parser.add_option("--target_tokens", type="int", dest="target_tokens")
    parser.add_option("--data_source_bins", type="str", dest="data_source_bins")
    parser.add_option("--mix_file", type="str", dest="mix_file")
    parser.add_option("--output_prefix", type="str", dest="output_prefix")
    
    (options, args) = parser.parse_args()

    return options
        
def check_chat_template(entry, print_entry = False):
    try:
        chat = TOKENIZER.apply_chat_template(entry, tokenize=False)
        if print_entry:
            print(chat)
        return True
    except Exception as e:
        print(f'ERROR: Failed to tokenize entry: {e}')
        print(entry)
        # sys.exit()
        return False

def insert_system_prompt(entry):
    if entry['conversation'][0]['role'] == 'system':
        if not entry['conversation'][0].get('content', False):
            entry['conversation'][0]['content'] = SYSTEM
    else:
        entry['conversation'].insert(0, {'role': 'system', 'content': SYSTEM})
    return entry

def process_entry(entry, reasoning_effort, source_name):

    ## Add self-identity system prompt
    if source_name in ['self-identity', 'self-identity-extended', 'uae-safety', 'nemotron-safety', 'adversarial-safety']:
        do_insert_system_prompt = random.random() < 0.5
    else:
        do_insert_system_prompt = random.random() < 0.1

    if do_insert_system_prompt:
        entry = insert_system_prompt(entry)

    entry_tokens = entry['token_count_answer'] + entry['token_count_think']

    reasoning_effort_tokens = {
        'low': 'think_faster',
        'medium': 'think_fast',
        'high': 'think',
    }

    ## Handle multi-turn entries - we split each entry into multiple entries, one for each turn
    conversation = entry['conversation']
    turn_count = sum([1 for turn in conversation if turn['role'] == 'assistant'])
    entry_data = []

    for i in range(turn_count):
        conversation_i = []
        turn_count_i = 0
        for turn in conversation:
            if turn['role'] != 'assistant':
                conversation_i.append(turn)
            else:

                has_tools = "tool_calls" in turn and turn['tool_calls']
                has_content = 'content' in turn and turn['content']

                if not (has_tools or has_content):
                    print('ERROR: Assistant response has no tools or content when processing entry - THIS SHOULD NEVER HAPPEN')
                    print(turn)
                    # sys.exit()

                elif turn_count_i == i: ## Add assistant response with think tokens and break

                    current_entry = conversation_i.copy()
                    
                    current_turn = {'role': 'assistant'}

                    if 'think' in turn:
                        current_turn[reasoning_effort_tokens[reasoning_effort]] = turn['think']
                    else:
                        current_turn[reasoning_effort_tokens[reasoning_effort]] = ''
                        if entry['token_count_think'] > 0:
                            print('WARNING: think is missing for turn', i)

                    if has_content:
                        current_turn['content'] = turn['content']
                    if has_tools:
                        current_turn['tool_calls'] = turn['tool_calls']
                    
                    current_entry.append(current_turn)
                    
                    if check_chat_template(current_entry):
                        entry_data.append(current_entry)
                    break

                else: ## Add assistant response without think tokens and continue
                    assistant_turn = {'role': 'assistant'}
                    if has_content:
                        assistant_turn['content'] = turn['content']
                    if has_tools:
                        assistant_turn['tool_calls'] = turn['tool_calls']
                    conversation_i.append(assistant_turn)
                    turn_count_i += 1

    return entry_data, entry_tokens

def main():

    options = parse_args()
    print(options)

    target_tokens = options.target_tokens
    data_source_bins = options.data_source_bins
    mix_file = options.mix_file
    output_prefix = options.output_prefix

    output_directory = 'YOUR_OUTPUT_DIRECTORY'
    
    random.seed(17)

    with open(data_source_bins, "r") as f:
        data_source_bins = json.load(f)

    with open(mix_file, "r") as f:
        mix = json.load(f)

    ## Check if mix is valid
    mix_weights = [s.get('weight') for s in mix if isinstance(s, dict)]
    mix_names = [s.get('name') for s in mix if isinstance(s, dict)]
    mix_reasoning_effort = [s.get('reasoning_effort') for s in mix if isinstance(s, dict)]

    if abs(sum([w for w in mix_weights if w > 0]) - 1.0) > 1e-4:
        sys.exit('ERROR: Mix weights do not sum to 1.0')
    else:
        print('Total weight is 1.0')

    source_names = [s.get('name') for s in data_source_bins if isinstance(s, dict)]
    if len(source_names) != len(set(source_names)):
        sys.exit('ERROR: Source names are not unique')

    source_info = [] # [name, path, token count, reasoning effort]
    for i, name in enumerate(mix_names):
        if name not in source_names:
            sys.exit(f'ERROR: Source not found in data_source_bins: {name}')
        else:
            if mix_weights[i] != 0:
                if mix_weights[i] > 0:
                    target_tokens_source = int(mix_weights[i] * target_tokens)
                else:
                    target_tokens_source = mix_weights[i]
                source_info.append([name, data_source_bins[source_names.index(name)].get('path'), target_tokens_source, mix_reasoning_effort[i]])
            else:
                print('Skipping source', name, 'with weight 0')

    print('MIX IS VALID')
    output_tmp_directory = output_prefix + '_tmp'
    os.makedirs(output_tmp_directory, exist_ok=True)
    print(source_info)
    print('--------------------------------\n')
    sys.stdout.flush()
    total_tokens = 0

    ## Our dataset [https://huggingface.co/datasets/LLM360/TxT360-3efforts] is already sampled. Below code is just for reference.
    for name, path, target_tokens_source, reasoning_effort in source_info:

        print(f'Processing {name}... with reasoning effort {reasoning_effort}; collecting {target_tokens_source} tokens')
        if os.path.exists(os.path.join(output_tmp_directory, f'{name}.jsonl')):
            print(f'{name}-{reasoning_effort} already collected; skipping\n')
            continue
        
        if os.path.exists(os.path.join(output_directory, f'{name}.jsonl')):
            print(f'Found {name} in processed keep directory')
        else:
            print(f'ERROR: {name} is not processed yet - run process_oss_sources.py first')
            sys.exit()

        time_s = time.time()

        source_answer_tokens = 0
        source_think_tokens = 0
        source_data = []

        with open(os.path.join(output_directory, f'{name}.jsonl'), "r") as f:
            for line in f:
                entry = json.loads(line)
                source_answer_tokens += entry['token_count_answer']
                source_think_tokens += entry['token_count_think']
                source_data.append(entry)

        # Print stats
        print(f'Total entries: {len(source_data)}')
        print(f'Total answer tokens (B): {source_answer_tokens/1e9}')
        print(f'Total think tokens (B): {source_think_tokens/1e9}')
        print(f'Total tokens (B): {(source_answer_tokens + source_think_tokens)/1e9}')
        print(f'Average answer tokens per entry: {source_answer_tokens / len(source_data)}')
        print(f'Average think tokens per entry: {source_think_tokens / len(source_data)}')
        print(f'Average total tokens per entry: {(source_answer_tokens + source_think_tokens) / len(source_data)}')
        sys.stdout.flush()

        # Negative weights correspond to copying the dataset
        if target_tokens_source < 0:
            target_tokens_source = abs(target_tokens_source) * (source_answer_tokens + source_think_tokens)

        # If source is small, we need to sample more entries
        if source_answer_tokens + source_think_tokens < target_tokens_source:
            print(f'WARNING: Source {name}-{reasoning_effort} is too small, sampling more entries')
            # Sample more entries
            avg_tokens_per_entry = (source_answer_tokens + source_think_tokens) / len(source_data)
            n_data_duplicates = target_tokens_source // (source_answer_tokens + source_think_tokens)
            num_additional_entries = int((target_tokens_source % (source_answer_tokens + source_think_tokens)) / avg_tokens_per_entry)
            additional_entries = random.sample(source_data, num_additional_entries)
            source_data = n_data_duplicates * source_data + additional_entries
            print(f'Duplicated {n_data_duplicates} times and sampled {num_additional_entries} additional entries; Total entries: {len(source_data)}')
            random.shuffle(source_data)
            print(f'Shuffled {len(source_data)} entries...')
        else:
            print(f'OK: Source {name}-{reasoning_effort} has enough tokens: {source_answer_tokens + source_think_tokens} tokens; target is {target_tokens_source} tokens')

        source_tokens_collected = 0
        source_data_sampled = []
        for entry in source_data:
            try:
                entry_data, entry_tokens = process_entry(entry, reasoning_effort, name)
            except:
                print(f'ERROR: Failed to process entry at collection - THIS SHOULD NEVER HAPPEN')
                print(entry)
                continue
            source_tokens_collected += entry_tokens
            total_tokens += entry_tokens
            source_data_sampled.extend(entry_data)
            if source_tokens_collected >= target_tokens_source:
                print(f'SUCCESS: Collected {source_tokens_collected} tokens from {name}-{reasoning_effort} with target {target_tokens_source}')
                break
        if source_tokens_collected < target_tokens_source:
            print(f'WARNING: Found less tokens than expected in source {name}-{reasoning_effort}: found {source_tokens_collected} tokens, expected {target_tokens_source} tokens')

        print(f'Time taken: {time.time() - time_s} seconds')

        output_source_file = os.path.join(output_tmp_directory, f'{name}_{reasoning_effort}.jsonl')
        with open(output_source_file, "w") as f:
            for entry in source_data_sampled:
                f.write(json.dumps(entry) + "\n")
        print(f'SUCCESS: Saved {name}-{reasoning_effort} {len(source_data_sampled)} SAMPLED DOCUMENTS to {output_source_file}')

        print(f'EXAMPLE ENTRY FROM {name}-{reasoning_effort}:')
        check_chat_template(source_data_sampled[-1], print_entry = True)
        print('--------------------------------\n')
        sys.stdout.flush()

    ## Merge all tmp files into one
    ## You can directly run merging on TxT360-3efforts
    print('Merging all tmp files into one...')
    all_tmp_files = [os.path.join(output_tmp_directory, f) for f in os.listdir(output_tmp_directory) if f.endswith('.jsonl')]
    ALL_DATA = []
    for file in all_tmp_files:
        with open(file, "r") as f_in:
            for line in f_in:
                ALL_DATA.append(line.rstrip('\n'))

    print('\nMIX STATISTICS:')
    print(f'Total number of documents: {len(ALL_DATA)}')
    print(f'Total token count: {total_tokens}')
    print(f'Average tokens per document: {total_tokens / len(ALL_DATA)}')

    random.shuffle(ALL_DATA)
    with open(output_prefix + '.jsonl', "w") as f:
        for entry in ALL_DATA:
            f.write(entry + '\n')

    print(f'SUCCESS: Saved {len(ALL_DATA)} documents to {output_prefix + '.jsonl'}')

if __name__ == '__main__':
    main()