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

    def get_function_call(self, message):
        func = message.split("FUNCTION_CALL")[-1]
        func = func.strip(" \n")
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"
        if func.startswith("['"):
            func = func.replace("['", "[")
        if func.endswith("]'"):
            func = func.replace("]'", "]")
        return func

    def parse_function_call(self, function_call):
        name = function_call.split("(")[0][1:]
        args = function_call.split("(")[1][:-2]
        return name, args


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
                    "content": turn["request"]["api_name"] + json.dumps(turn["request"]["parameters"])
                    })

                # Tool response
                response_content = {
                    "response": turn["response"],
                    "exception": turn["exception"]
                }
                openai_history.append({
                    "role": "user",
                    "content": json.dumps(response_content)
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

        if "FUNCTION_CALL" in openai_message:
            function_call = self.get_function_call(openai_message)
            api_name, parameters_str = self.parse_function_call(function_call)
            try:
                parameters = json.loads(parameters_str)
            except json.decoder.JSONDecodeError:
                # check termination reason
                logger.info(f"Failed to decode arguments for {api_name}: {parameters_str}")
                parameters = None
            return {
                "role": "api",
                "request": {
                    "api_name": api_name,
                    "parameters": parameters
                },
                "metadata": metadata,
            }
        else:
            return {
                "role": "assistant",
                "text": openai_message,
                "metadata": metadata,
            }
