import os
import json
import argparse
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


REQUIRED_TOPICS = [
    'person_names',
    'email_addresses',
    'phone_numbers',
    'addresses',
    'national_ids'
]


def create_client() -> OpenAI:
    """
    Creates and returns an OpenAI client connected to a local vLLM server.
    """
    return OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="apikeynotrequired"
    )


def read_txt_into_batches(txt_path: str, batch_size: int) -> List[str]:
    """
    Reads a text file and splits its content into clean batches.
    """
    with open(txt_path, 'r', encoding='utf-8') as file:
        input_text = file.read()
    input_text_clean = ' '.join(input_text.strip().split())
    return [input_text_clean[i:i + batch_size] for i in range(0, len(input_text_clean), batch_size)]


def read_system_prompt_and_text_inputs(system_prompt_path: str, input_path: str) -> Tuple[str, List[Dict]]:
    """
    Reads the system prompt and list of input text file paths. 
    Splits each text input into batches.
    """
    with open(system_prompt_path, 'r', encoding='utf-8') as file:
        system_prompt = file.read()

    with open(input_path, 'r', encoding='utf-8') as file:
        input_paths = list(set(file.read().splitlines()))

    all_inputs = [{
        'file_path': path,
        'batches': read_txt_into_batches(path, batch_size=4000)
    } for path in input_paths if path.strip()]

    return system_prompt, all_inputs


def get_response(text: str, client: OpenAI, model_path: str, system_prompt: str) -> str:
    """
    Sends a single text batch to the LLM and retrieves the JSON response.
    """
    completion = client.chat.completions.create(
        model=model_path,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        response_format={'type': "json_object"}
    )
    return completion.choices[0].message.content


def get_response_multithread(
    batches: List[str],
    client: OpenAI,
    model_path: str,
    system_prompt: str,
    max_parallel_requests: int = 20
) -> List[str]:
    """
    Sends multiple text batches to the LLM in parallel and returns the responses.
    """
    responses = []
    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        futures = {
            executor.submit(get_response, text, client, model_path, system_prompt): text
            for text in batches
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Samples"):
            try:
                result = future.result()
                if result is not None:
                    responses.append(result)
            except Exception as e:
                print(f"Error during response: {e}")

    return responses


def eval_llm_responses(llm_responses: List[str]) -> List[Dict]:
    """
    Converts LLM response strings into Python objects (lists of dicts).
    """
    results = []
    for response in tqdm(llm_responses):
        try:
            results.append(eval(response))
        except Exception:
            print("Error parsing LLM response. Skipping.")
    return results


def merge_llm_responses(llm_responses_eval: List[Dict], required_topics: List[str]) -> Dict[str, List]:
    """
    Merges all parsed LLM responses into a single dictionary grouped by topic.
    """
    merged = {topic: [] for topic in required_topics}
    for response in tqdm(llm_responses_eval):
        for topic in required_topics:
            try:
                merged[topic].extend(response.get(topic, []))
            except Exception:
                print("Error merging topic from batch. Skipping.")
    return merged


def get_llm_outputs_and_save_results(
    output_path: str,
    all_inputs: List[Dict],
    client: OpenAI,
    model_path: str,
    system_prompt: str
) -> None:
    """
    For each input text file, gets LLM outputs and saves results to JSON files.
    """
    os.makedirs(output_path, exist_ok=True)

    for input_entry in tqdm(all_inputs, desc="Processing Files"):
        try:
            batches = input_entry['batches']
            file_path = input_entry['file_path']

            llm_responses = get_response_multithread(
                batches, client, model_path, system_prompt
            )
            parsed = eval_llm_responses(llm_responses)
            merged = merge_llm_responses(parsed, REQUIRED_TOPICS)

            save_name = os.path.join(
                output_path,
                file_path.rsplit('.', 1)[0].replace('/', '|') + '.json'
            )

            with open(save_name, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=4, ensure_ascii=False)

            print(f"\nSaved to {save_name}")
        except Exception as e:
            print(f"Error processing file {input_entry['file_path']}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM safety check and extract structured data.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--system-prompt-path", type=str, required=True, help="Path to the system prompt file.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the file containing input .txt paths.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save JSON output.")
    return parser.parse_args()


def main():
    args = parse_args()

    client = create_client()
    system_prompt, all_inputs = read_system_prompt_and_text_inputs(
        args.system_prompt_path,
        args.input_path
    )

    print(f"{len(all_inputs)} input files given")
    get_llm_outputs_and_save_results(
        args.output_path,
        all_inputs,
        client,
        args.model_path,
        system_prompt
    )


if __name__ == "__main__":
    main()
