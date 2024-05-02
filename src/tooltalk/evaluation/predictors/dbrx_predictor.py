import json
import logging

from tooltalk.evaluation.tool_executor import BaseAPIPredictor
from transformers import AutoTokenizer

from .constants import SYSTEM_PROMPT

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

    def predict(self, metadata: dict, conversation_history: dict) -> dict:
        api_docs_str = '\n'.join([json.dumps(api_doc) for api_doc in self.api_docs])
        system_prompt = SYSTEM_PROMPT.format(
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
        # metadata = {
        metadata = {}
        if "function_call" in openai_message:
            function_call = openai_message["function_call"]
            api_name = function_call["name"]
            try:
                parameters = json.loads(function_call["arguments"])
            except json.decoder.JSONDecodeError:
                # check termination reason
                logger.info(f"Failed to decode arguments for {api_name}: {function_call['arguments']}")
                parameters = None
            return {
                "role": "api",
                "request": {
                    "api_name": api_name,
                    "parameters": parameters
                },
                # store metadata about call
                "metadata": metadata,
            }
        else:
            return {
                "role": "assistant",
                "text": openai_message,
                # store metadata about call
                "metadata": metadata,
            }
