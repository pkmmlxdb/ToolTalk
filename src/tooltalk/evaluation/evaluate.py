"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Evaluate Tool LLM on API-Talk dataset.
"""
import argparse
import json
import logging
import os
from collections import Counter
from enum import Enum
from typing import List

import openai
from tooltalk.apis import ALL_APIS, APIS_BY_NAME, SUITES_BY_NAME
from tooltalk.evaluation.tool_executor import BaseAPIPredictor, ToolExecutor
from tooltalk.utils.file_utils import get_names_and_paths
from tqdm import tqdm
from transformers import AutoTokenizer

from .predictors.guided_predictor import GuidedPredictor
from .predictors.mistral_predictor import MistralPredictor
from .predictors.unguided_predictor import UnguidedPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalModes(str, Enum):
    PREDICT = "predict"
    EVALUATE = "evaluate"
    VALIDATE = "validate"


def get_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, help="Path to dataset for models to evaluate")
    parser.add_argument("--database", type=str, help="Path to database used in evaluation")
    parser.add_argument("--api_key", type=str, default="openai.key", help="Path to OpenAI API key")
    parser.add_argument("--base_url", type=str, help="The base url for the API.")
    parser.add_argument("--api_mode", type=str, choices=["exact", "suite", "all"], default="all",
                        help="API mode to use for evaluation, determines which api docs to include")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model to use for generation")
    parser.add_argument("--output_dir", type=str, help="Path to output model predictions")
    parser.add_argument("--reset", action="store_true", help="reset evaluation writing over any cached results")
    parser.add_argument("--disable_documentation", action="store_true",
                        help="disabled documentation sent to GPT-4 replacing with empty strings")
    parser.add_argument("--modes", choices=list(EvalModes), type=str, nargs='+', default=list(EvalModes),
                        help="Evaluation modes")
    parser.add_argument("--predictor", type=str, help="The model predictor")
    parser.add_argument("--guided", action="store_true", default=False, help="Use guided generation.")

    return parser


def main(flags: List[str] = None):
    parser = get_arg_parser()
    args = parser.parse_args(flags)

    # Initialize OpenAI client
    openai_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_key is None:
        openai_key = args.api_key
    client = OpenAI(
        api_key=openai_key,
        base_url=args.base_url,
    )

    total_metrics = Counter()
    os.makedirs(args.output_dir, exist_ok=True)
    tool_executor = ToolExecutor(init_database_dir=args.database)
    for file_name, file_path in tqdm(get_names_and_paths(args.dataset)):
        output_file_path = os.path.join(args.output_dir, file_name)
        if os.path.exists(output_file_path) and not args.reset:
            logger.info(f"Skipping {file_name} because it already exists")
            with open(output_file_path, 'r', encoding='utf-8') as reader:
                conversation_with_metrics = json.load(reader)
            total_metrics += conversation_with_metrics["metrics"]
            total_metrics["num_conversations"] += 1
            continue

        logger.info(f"Running {file_name}")
        with open(file_path, 'r', encoding='utf-8') as reader:
            conversation = json.load(reader)

        if EvalModes.PREDICT in args.modes:
            logger.info("Running prediction...")
            if args.api_mode == "exact":
                apis_used = [APIS_BY_NAME[api_name] for api_name in conversation["apis_used"]]
            elif args.api_mode == "suite":
                apis_used = [
                    api for suite_name in conversation["suites_used"] for api in SUITES_BY_NAME[suite_name].apis
                ]
            elif args.api_mode == "all":
                apis_used = ALL_APIS
            else:
                raise ValueError(f"Invalid api mode: {args.api_mode}")


            if args.guided:
                 predictor_func = GuidedPredictor(
                    client=client,
                    model=args.model,
                    apis_used=apis_used,
                    disable_docs=args.disable_documentation
                )
            else:
                predictor_func = UnguidedPredictor(
                    client=client,
                    model=args.model,
                    apis_used=apis_used,
                    disable_docs=args.disable_documentation
                )

            # if args.predictor == "dbrx":
                # predictor_func = DBRXPredictor(
                #     client=client,
                #     model=args.model,
                #     apis_used=apis_used,
                #     disable_docs=args.disable_documentation
                # )
            # elif args.predictor == "openai":
            #     predictor_func = OpenAIPredictor(
            #         client=client,
            #         model=args.model,
            #         apis_used=apis_used,
            #         disable_docs=args.disable_documentation
            #     )
            # elif args.predictor == "mistral":
            #     predictor_func = MistralPredictor(
            #         client=client,
            #         model=args.model,
            #         apis_used=apis_used,
            #         disable_docs=args.disable_documentation
            #     )

            conversation = tool_executor.run_conversation(conversation, predictor_func)

        if EvalModes.EVALUATE in args.modes:
            logger.info("Running evaluation...")
            conversation = tool_executor.evaluate_predictions(conversation)
            print(conversation)
            logger.info(f"Conversation {file_name} pass: {conversation['metrics']['success']}")
            total_metrics += conversation["metrics"]
            total_metrics["num_conversations"] += 1

            if EvalModes.VALIDATE in args.modes:
                logger.info("Validating evaluation...")
                for turn in conversation["conversation"]:
                    if "predictions" not in turn:
                        continue
                    for prediction in turn["predictions"]:
                        if prediction["role"] == "api":
                            assert "match" in prediction
                            assert "bad_action" in prediction

        with open(output_file_path, 'w', encoding='utf-8') as writer:
            json.dump(conversation, writer, indent=4)

    logger.info("Finished processing conversations")
    if EvalModes.EVALUATE in args.modes:
        metrics = {
            "num_conversations": total_metrics["num_conversations"],
            "precision": total_metrics["matches"] / total_metrics["predictions"],
            "recall": total_metrics["matches"] / total_metrics["ground_truths"],
            "action_precision": total_metrics["valid_actions"] / total_metrics["actions"],
            "bad_action_rate": total_metrics["bad_actions"] / total_metrics["actions"],
            "success_rate": total_metrics["success"] / total_metrics["num_conversations"]
        }
        logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()
