import pickle
import transformers
import torch
import gc
import os
import datetime
import json
import logging


# ─── Constants ────────────────────────────────────────────────────────────────
CURRENT = 0  # 0-ja, 1-en, 2-zh
lang_dict = {0: 'ja', 1: 'en', 2: 'zh'}

repeat_time = 200
all_available_num = [400]
OUTPUT_DIR = "generated"
d_map = "cuda:2"

all_models_dict = {
    'llmjp': 'llm-jp/llm-jp-3-13b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen': 'Qwen/Qwen2.5-14B-Instruct',
    'glm': 'THUDM/glm-4-9b-chat',
    'phi': 'microsoft/phi-4'
}


# ─── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_dir=OUTPUT_DIR):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'data_generation_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# ─── Model Class ──────────────────────────────────────────────────────────────
class LLMs(object):
    def __init__(self, model_id, device_map):
        self.model_id = model_id
        self.device_map = device_map
        self.logger = logging.getLogger(__name__)
        self.pipeline = self.load_model()

    def load_model(self):
        self.logger.info(f"Loading model: {self.model_id}")
        return transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.device_map,
        )

    def text_generation(self, role: str, prompt_list: list[str], repeat: int = repeat_time, max_new_tokens: int = 1200) -> list[dict]:
        self.logger.info(f"Starting text generation - Language: {lang_dict[CURRENT]}, Prompts: {len(prompt_list)}, Repeats: {repeat}")
        response_list = []

        total_iterations = repeat * len(prompt_list)
        completed = 0

        for i in range(repeat):
            for prompt_idx, prompt in enumerate(prompt_list):
                messages = [
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt},
                ]

                outputs = self.pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                    top_p=0.8,
                    repetition_penalty=1.1,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id,
                    do_sample=True
                )

                response_dict = {
                    'prompt_idx': prompt_idx,
                    'repeat_idx': i,
                    'content': outputs[0]["generated_text"][-1]
                }
                response_list.append(response_dict)

                completed += 1
                if completed % 10 == 0:
                    self.logger.info(f"Progress: {completed}/{total_iterations} ({completed/total_iterations*100:.1f}%)")

        self.logger.info(f"Text generation completed: {total_iterations} responses generated")
        return response_list

    def generate_and_save(self, dest_file: str, role: str, prompt_list: list[str]):
        self.logger.info(f"Generating and saving to: {dest_file}")

        if os.path.exists(dest_file):
            self.logger.warning(f"File already exists, will be overwritten: {dest_file}")
            with open(dest_file, 'rb') as f1:
                response_dict = pickle.load(f1)
        else:
            response_dict = {}

        response_dict[CURRENT] = self.text_generation(role, prompt_list)

        with open(dest_file, 'wb') as f2:
            pickle.dump(response_dict, f2)

        self.logger.info(f"Successfully saved to: {dest_file}")

    def empty_cache(self):
        self.logger.info("Clearing GPU cache")
        if self.pipeline is not None:
            del self.pipeline
            gc.collect()
            torch.cuda.empty_cache()
        self.logger.info("GPU cache cleared")


# ─── Helper Functions ─────────────────────────────────────────────────────────
def add_limit_suffix(lang: int, prompt_list: list[str], available_num: int) -> list[str]:
    """Add character limit suffix to prompts."""
    limit_suffix_dict = {
        0: f'ストーリーを{available_num}字以内で完成させる。',
        1: f'Complete the story within {available_num} characters.',
        2: f'在{available_num}字以内完成故事。'
    }

    suffix = limit_suffix_dict[lang]
    return [prompt + suffix for prompt in prompt_list]


def generate_dimension_responses(model_name: str, model_id: str, dimension: str,
                                  sys_role: str, prompt_list: list[str]):
    """Generate responses for a specific dimension (PDI, IDV, etc.)."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing dimension: {dimension} for model: {model_name}")

    pipeline = LLMs(model_id=model_id, device_map=d_map)
    now_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')

    for available_num in all_available_num:
        suffixed_prompt_list = add_limit_suffix(CURRENT, prompt_list, available_num)
        suffixed_file = f'generated/{model_name}_{dimension}_{lang_dict[CURRENT]}_{available_num}_{repeat_time}_{now_str}.pkl'
        logger.info(f"Generating limited responses ({available_num} chars)...")
        pipeline.generate_and_save(suffixed_file, sys_role, suffixed_prompt_list)

    pipeline.empty_cache()
    logger.info(f"Completed dimension: {dimension}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging()

    # Load prompts and system role
    system_role_ja = 'あなたは、社会的な礼儀、尊重、人間関係の細かなニュアンスに深い理解を持つ熟練した物語作家です。与えられたプロンプトを基に物語を続け、登場人物の行動、感情、会話を通して、異なる文化的態度がどのように自然に人間関係に影響するかを示してください。ホフステードの文化次元理論に基づき、文化的視点の微妙な表現を描きつつも、文化差異を直接的に説明しないようにしてください。物語の続きが日本語、英語、または中国語の文化環境での一般的な態度や対人関係の細部を反映し、自然で流れるような展開になるように工夫してください。'

    with open('cultural_prompts_ja.json', 'r') as f:
        prompt_list_ja = json.load(f)

    sys_role_list = [system_role_ja]
    prompts_list = [prompt_list_ja]

    current_prompt_dict = prompts_list[CURRENT]
    current_sys_role = sys_role_list[CURRENT]

    # Start generation
    logger.info("=" * 70)
    logger.info("Starting data generation process")
    logger.info(f"Language: {lang_dict[CURRENT]}")
    logger.info(f"Repeat time: {repeat_time}")
    logger.info(f"Total dimensions: {len(current_prompt_dict)}")
    logger.info(f"Total models: {len(all_models_dict)}")
    logger.info("=" * 70)

    for model_idx, (model_name, model_id) in enumerate(all_models_dict.items(), 1):
        logger.info("=" * 70)
        logger.info(f"Processing model {model_idx}/{len(all_models_dict)}: {model_name}")
        logger.info("=" * 70)

        for dim_idx, (dimension, prompts) in enumerate(current_prompt_dict.items(), 1):
            logger.info("-" * 70)
            logger.info(f"Dimension {dim_idx}/{len(current_prompt_dict)}: {dimension}")

            try:
                generate_dimension_responses(
                    model_name=model_name,
                    model_id=model_id,
                    dimension=dimension,
                    sys_role=current_sys_role,
                    prompt_list=prompts
                )
            except Exception as e:
                logger.error(f"Error processing {model_name} - {dimension}: {str(e)}", exc_info=True)
                continue

        logger.info(f"✓ Completed all dimensions for {model_name}")
        logger.info("")

    logger.info("=" * 70)
    logger.info("Data generation process completed successfully")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()