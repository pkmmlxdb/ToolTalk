import json
import logging

from tooltalk.evaluation.tool_executor import BaseAPIPredictor

from .constants import SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIPredictor(BaseAPIPredictor):
    system_prompt = "You are a helpful assistant. Here is some user data:" \
                    "\nlocation: {location}" \
                    "\ntimestamp: {timestamp}" \
                    "\nusername (if logged in): {username}"

    def __init__(self, client, model, apis_used, disable_docs=False):
        self.model = model
        self.client = client
        self.api_docs = [{"type": "function", "function": api.to_openai_doc(disable_docs)} for api in apis_used]

    def predict(self, metadata: dict, conversation_history: dict) -> dict:
        system_prompt = self.system_prompt.format(
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


            # we currently use a function_call keyword, but this should probably be tool_call which has the following format
            # ChatCompletion(id='chatcmpl-9KXAPvtDiRwbHyFppflrv1suCkp7J', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_9JoDdZB4T8WomLfpC5dHdEJ8', function=Function(arguments='{"receiver":"ickyvicky","message":"Check out this video: utube.com/slkd"}', name='SendMessage'), type='function')]))], created=1714681081, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint='fp_3b956da36b', usage=CompletionUsage(completion_tokens=28, prompt_tokens=2194, total_tokens=2222))
            elif turn["role"] == "api":
                openai_history.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": { # should this be `tool_calls`? this will have a diff format
                        "name": turn["request"]["api_name"],
                        "arguments": json.dumps(turn["request"]["parameters"])
                    }
                })
                response_content = {
                    "response": turn["response"],
                    "exception": turn["exception"]
                }
                openai_history.append({
                    "role": "function",
                    "name": turn["request"]["api_name"],
                    "content": json.dumps(response_content)
                })

        openai_response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_history,
            tools=self.api_docs,
            tool_choice="auto",
            )

        logger.debug(f"OpenAI full response: {openai_response}")
        openai_message = openai_response.choices[0].message
        # metadata = {
        #     "openai_request": {
        #         "model": self.model,
        #         "messages": openai_history,
        #         "functions": self.api_docs,
        #     },
        #     "openai_response": openai_response
        # }
        metadata = {}
        if openai_message.tool_calls:
            function_call = openai_message.tool_calls[0].function
            api_name = function_call.name
            try:
                parameters = json.loads(function_call.arguments)
            except json.decoder.JSONDecodeError:
                # check termination reason
                logger.info(f"Failed to decode arguments for {api_name}: {function_call.arguments}")
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
                "text": openai_message.content,
                # store metadata about call
                "metadata": metadata,
            }

