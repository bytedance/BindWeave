 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import json
import base64
import openai
import traceback


class ArkClient:
    def __init__(self, model_name, api_key, base_url,
                 system_instruct=None, max_retries=2, timeout=60):
        if not isinstance(api_key, list):
            api_key = [api_key]
        self._client_list = [openai.OpenAI(
            api_key=key, base_url=base_url, max_retries=max_retries) for key in api_key]
        self._model_name = model_name
        self._idx = 0
        self._system_instruct = system_instruct
        self._timeout = timeout

    def gpt_call(self, messages, max_tokens, timeout, extra_body=None):
        target = self._client_list[self._idx]
        self._idx = (self._idx + 1) % len(self._client_list)

        completion = target.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body
        )
        return completion

    def predict(self, prompt, image_list=None, max_tokens=2000, timeout=None, extra_body=None):
        timeout = timeout if timeout is not None else self._timeout
        content = []
        elements = prompt.split('<image>')

        if image_list is not None:
            assert len(elements) == len(image_list) + \
                1, '{} != {}'.format(len(elements), len(image_list) + 1)

        for i, element in enumerate(elements[:-1]):
            if len(element) > 0:
                content.append({'type': 'text', 'text': element})
            image = image_list[i]
            image = base64.b64encode(image).decode('utf8')
            # content.append({'type': 'image', 'image': image, "resize": 768})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })
        if len(elements[-1]) > 0:
            content.append({'type': 'text', 'text': elements[-1]})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        if self._system_instruct is not None:
            messages = [
                {"role": "system", "content": self._system_instruct}] + messages

        rsp = self.gpt_call(messages, max_tokens=max_tokens, timeout=timeout, extra_body=extra_body)

        rsp = rsp.model_dump_json()
        rsp = json.loads(rsp)

        # print('usage', rsp['usage'])
        assert rsp['usage']['completion_tokens'] < max_tokens, 'gpt_call: incomplete generation'

        content = rsp['choices'][0]['message']['content']

        # print(prompt,'\n',content)

        ret = {'content': content, 'usage': rsp['usage'], 'prompt': prompt}
        if 'reasoning_content' in rsp['choices'][0]['message']:
            ret['reasoning_content'] = rsp['choices'][0]['message']['reasoning_content']
            # print('reasoning_content', ret['reasoning_content'])

        return ret

