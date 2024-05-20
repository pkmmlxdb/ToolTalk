import json
import logging
import os
from openai import OpenAI

from eval_generations.benchmarks.ToolTalk.src.tooltalk.evaluation.tool_executor import BaseAPIPredictor
from transformers import AutoTokenizer

from .constants import SYSTEM_PROMPT, DBRX_SYSTEM_PROMPT

from eval_generations.utils.vllm import VLLM_Generator
from eval_generations.utils.generator import call_generate, Generator, Prompt
from eval_generations.defaults import PROJECT_PATH
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnguidedPredictor(BaseAPIPredictor):

    def __init__(self, model, apis_used, argslist):
        if argslist.is_openai:
            # Initialize OpenAI client
            openai_key = os.environ.get("OPENAI_API_KEY", None)
            if openai_key is None:
                openai_key = argslist.api_key
            self.client = OpenAI(
                api_key=openai_key,
                base_url=argslist.base_url,
            )
        else: 
            run_name = model.split('/')[1].lower() + '-' + argslist.dataset.split('/')[-1].lower()

            # We will use a vLLM wrapper 
            self.model = VLLM_Generator(
                model_path_or_name=model,
                max_minutes_to_wait_for_vllm=120,
                deployment_id=run_name,
                go_to_11=True,
                remote=True,
                stop_deployment=False,
                clusters_not_in=['r14z4']
            ).__enter__()
#            self.model.__enter__()
        
        self.temperature = 0 
        self.top_p = 1
        self.max_tokens = 4096

        self.is_openai = argslist.is_openai
        self.api_docs = [api.to_openai_doc(argslist.disable_documentation) for api in apis_used]

    def get_function_call(self, message, start_tag = "<tool_call>", end_tag = "</tool_call>"):

        start_index = message.find(start_tag)
        if start_index == -1:
            return None

        end_index = message.find(end_tag, start_index + len(start_tag))
        if end_index == -1:
            # if end_tag not present, check for last occurrence of start_tag and use this as the end_index
            end_index = message.rfind(start_tag, start_index + len(start_tag))
            if end_index == -1:
                return None

        function_call = message[start_index + len(start_tag):end_index].strip(" \n")
        return function_call

    def is_tool_used(self, message):
        return "<tool_call>" in message and "</tool_call>" in message

    def predict(self, metadata: dict, conversation_history: dict) -> dict:
        api_docs_str = '\n'.join([json.dumps(api_doc) for api_doc in self.api_docs])
        system_prompt = SYSTEM_PROMPT.format(
            functions=api_docs_str,
            location=metadata["location"],
            timestamp=metadata["timestamp"],
            username=metadata.get("username")
        )

        conv_history = [{
            "role": "system",
            "content": system_prompt
        }]
        for turn in conversation_history:
            if turn["role"] == "user" or turn["role"] == "assistant":
                conv_history.append({
                    "role": turn["role"],
                    "content": turn["text"]
                })
            elif turn["role"] == "api":
                # Tool call
                conv_history.append({
                    "role": "assistant",
                    "content": json.dumps(turn["request"]["api_name"]) + json.dumps(turn["request"]["parameters"])
                    })

                # Tool response
                response_content = {
                    "response": turn["response"],
                    "exception": turn["exception"]
                }
                conv_history.append({
                    "role": "user",
                    "content": f"<tool_response>\n{json.dumps(response_content)}\n</tool_response>"
                })
        
        if self.is_openai: 
            model_response = self.client.chat.completions.create(
                model=self.argslist.model,
                prompt=conv_history,
                temperature=self.temperature,
                top_p = self.top_p,
                max_tokens=self.max_tokens,
                )
        else: 
            try:
                model_response = call_generate(self.model.generate,
                            Prompt(
                                id = None,
                                prompt=conv_history
                            ),
                            use_completion_endpoint=False,
                            with_retry=True,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_tokens=self.max_tokens,
                        )
            except Exception as e:
                print(e)
                import time; time.sleep(10); pass

        
        logger.debug(f"Model full response: {model_response}")
        model_message = model_response

        # Get metadata
        # metadata = {
        #     "openai_request": {
        #         "model": self.model,
        #         "messages": conv_history,
        #         "functions": self.api_docs,
        #     },
        #     "model_response": "" #model_response
        # }
        metadata = {
            "tokens": {
                # "completion_tokens": model_response.usage.completion_tokens,
                # "prompt_tokens": model_response.usage.prompt_tokens,
                # "total_tokens": model_response.usage.total_tokens,
                }
            }

        if not self.is_tool_used(model_message):
            return {
                "role": "assistant",
                "text": model_message,
                "metadata": metadata,
            }

        try:
            function_call_str = self.get_function_call(model_message)
        except TypeError:
            logger.info(f"Failed to decode the tags of this function call:\n{function_call_str}")
            parameters = None
            api_name = None
            return {
                "role": "api",
                "request": {
                    "api_name": api_name,
                    "parameters": parameters
                },
                "metadata": metadata,
            }

        try:
            function_call_str = self.get_function_call(model_message)
            function_call_json = json.loads(function_call_str)
            api_name, parameters = function_call_json["name"], function_call_json["arguments"]
        except json.decoder.JSONDecodeError:
            logger.info(f"Failed to decode this function call into json:\n{function_call_str}")
            parameters = None
            api_name = None

        return {
            "role": "api",
            "request": {
                "api_name": api_name,
                "parameters": parameters
            },
            "metadata": metadata,
        }
