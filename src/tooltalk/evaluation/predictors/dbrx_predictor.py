import json
import logging

from tooltalk.evaluation.tool_executor import BaseAPIPredictor
from transformers import AutoTokenizer

from .constants import DBRX_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DBRXPredictor(BaseAPIPredictor):

    def __init__(self, client, model, apis_used, disable_docs=False):
        self.client=client
        self.model = model
        self.api_docs = [api.to_openai_doc(disable_docs) for api in apis_used]
        self.tokenizer = AutoTokenizer.from_pretrained(
            "databricks/dbrx-instruct",
            trust_remote_code=True,
            token="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",
            )

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

        function_call = message[start_index + len(start_tag):end_index].strip()
        return function_call


    def parse_function_call(self, function_call):
        name = function_call.split("(")[0][1:]
        args = function_call.split("(")[1][:-2]
        return name, args

    def is_tool_used(self, message):
        return "<tool_call>" in message or "</tool_call>" in message

    def predict(self, metadata: dict, conversation_history: dict) -> dict:
        api_docs_str = '\n'.join([json.dumps(api_doc) for api_doc in self.api_docs])
        system_prompt = DBRX_SYSTEM_PROMPT.format(
            functions=api_docs_str,
            location=metadata["location"],
            timestamp=metadata["timestamp"],
            username=metadata.get("username")
        )

        openai_history = [{
            "role": "system",
            "content": system_prompt
        }]
        for turn in conversation_history:
            if turn["role"] == "user" or turn["role"] == "assistant":
                openai_history.append({
                    "role": turn["role"],
                    "content": turn["text"]
                })
            elif turn["role"] == "api":

                # Tool call
                openai_history.append({
                    "role": "assistant",
                    "content": json.dumps(turn["request"]["api_name"]) + json.dumps(turn["request"]["parameters"])
                    })

                # Tool response
                response_content = {
                    "response": turn["response"],
                    "exception": turn["exception"]
                }
                openai_history.append({
                    "role": "user",
                    "content": f"<tool_response>\n{json.dumps(response_content)}\n</tool_response>"
                })

        prompt = self.tokenizer.apply_chat_template(openai_history, tokenize=False, add_generation_prompt=True)
        openai_response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            extra_body={'use_raw_prompt': True},
            )
        logger.debug(f"OpenAI full response: {openai_response}")
        openai_message = openai_response.choices[0].text
        # metadata = {
        #     "openai_request": {
        #         "model": self.model,
        #         "messages": openai_history,
        #         "functions": self.api_docs,
        #     },
        #     "openai_response": "" #openai_response
        # }
        metadata = {}

        if not self.is_tool_used(openai_message):
            return {
                "role": "assistant",
                "text": openai_message,
                "metadata": metadata,
            }
        else:
            try:
                function_call_str = self.get_function_call(openai_message)
                function_call_json = json.loads(function_call_str)
                api_name, parameters = function_call_json["name"], function_call_json["arguments"]
            except json.decoder.JSONDecodeError:
                # check termination reason
                logger.info(f"Failed to decode this function call:\n{function_call_str}")
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
