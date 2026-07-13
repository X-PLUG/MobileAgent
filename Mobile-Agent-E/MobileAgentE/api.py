import base64
import requests
from time import sleep
import json


MINIMAX_PRICING = {
    "USD": {
        "MiniMax-M3": {
            "standard": {
                "short": {"input": 0.3, "output": 1.2, "cache_read": 0.06, "cache_write": None},
                "long": {"input": 0.6, "output": 2.4, "cache_read": 0.12, "cache_write": None},
            },
            "priority": {
                "short": {"input": 0.45, "output": 1.8, "cache_read": 0.09, "cache_write": None},
                "long": {"input": 0.9, "output": 3.6, "cache_read": 0.18, "cache_write": None},
            },
        },
        "MiniMax-M2.7": {"input": 0.3, "output": 1.2, "cache_read": 0.06, "cache_write": 0.375},
    },
    "CNY": {
        "MiniMax-M3": {
            "standard": {
                "short": {"input": 2.1, "output": 8.4, "cache_read": 0.42, "cache_write": None},
                "long": {"input": 4.2, "output": 16.8, "cache_read": 0.84, "cache_write": None},
            },
            "priority": {
                "short": {"input": 3.15, "output": 12.6, "cache_read": 0.63, "cache_write": None},
                "long": {"input": 6.3, "output": 25.2, "cache_read": 1.26, "cache_write": None},
            },
        },
        "MiniMax-M2.7": {"input": 2.1, "output": 8.4, "cache_read": 0.42, "cache_write": 2.625},
    },
}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def build_api_url(base_url, api_protocol):
    api_protocol = api_protocol.lower()
    if api_protocol not in ("openai", "anthropic"):
        raise ValueError(f"Unsupported API protocol: {api_protocol}")
    path = "/chat/completions" if api_protocol == "openai" else "/v1/messages"
    return f"{base_url.rstrip('/')}{path}"


def _minimax_rates(model, input_tokens, service_tier, price_currency):
    currency_pricing = MINIMAX_PRICING.get(price_currency)
    if currency_pricing is None or model not in currency_pricing:
        return None
    model_pricing = currency_pricing[model]
    if model == "MiniMax-M3":
        length_tier = "long" if input_tokens > 512000 else "short"
        return model_pricing[service_tier][length_tier]
    return model_pricing


def track_usage(res_json, api_key, service_tier="standard", price_currency="USD"):
    """
    {'id': 'chatcmpl-AbJIS3o0HMEW9CWtRjU43bu2Ccrdu', 'object': 'chat.completion', 'created': 1733455676, 'model': 'gpt-4o-2024-11-20', 'choices': [...], 'usage': {'prompt_tokens': 2731, 'completion_tokens': 235, 'total_tokens': 2966, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'system_fingerprint': 'fp_28935134ad'}
    """
    model = res_json['model']
    usage = res_json['usage']
    openai_usage = "prompt_tokens" in usage and "completion_tokens" in usage
    if openai_usage:
        prompt_tokens, completion_tokens = usage['prompt_tokens'], usage['completion_tokens']
    elif "promptTokens" in usage and "completionTokens" in usage:
        prompt_tokens, completion_tokens = usage['promptTokens'], usage['completionTokens']
    elif "input_tokens" in usage and "output_tokens" in usage:
        prompt_tokens, completion_tokens = usage['input_tokens'], usage['output_tokens']
    else:
        prompt_tokens, completion_tokens = None, None

    cache_read_tokens = usage.get("cache_read_input_tokens", 0)
    cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
    if openai_usage:
        cache_read_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    prompt_token_price = None
    completion_token_price = None
    cache_read_token_price = None
    cache_write_token_price = None
    selected_service_tier = res_json.get("service_tier") or service_tier or "standard"
    if selected_service_tier == "default":
        selected_service_tier = "standard"
    price_currency = price_currency.upper()
    if prompt_tokens is not None and completion_tokens is not None:
        if "gpt-4o" in model:
            prompt_token_price = (2.5 / 1000000) * prompt_tokens
            completion_token_price = (10 / 1000000) * completion_tokens
        elif "gemini" in model:
            prompt_token_price = (1.25 / 1000000) * prompt_tokens
            completion_token_price = (5 / 1000000) * completion_tokens
        elif "claude" in model:
            prompt_token_price = (3 / 1000000) * prompt_tokens
            completion_token_price = (15 / 1000000) * completion_tokens
        elif model in ("MiniMax-M3", "MiniMax-M2.7"):
            pricing_input_tokens = prompt_tokens
            billable_input_tokens = prompt_tokens
            if openai_usage:
                billable_input_tokens = max(prompt_tokens - cache_read_tokens, 0)
            else:
                pricing_input_tokens += cache_read_tokens + cache_write_tokens
            rates = _minimax_rates(model, pricing_input_tokens, selected_service_tier, price_currency)
            if rates is not None:
                prompt_token_price = (rates["input"] / 1000000) * billable_input_tokens
                completion_token_price = (rates["output"] / 1000000) * completion_tokens
                if rates["cache_read"] is not None:
                    cache_read_token_price = (rates["cache_read"] / 1000000) * cache_read_tokens
                if rates["cache_write"] is not None:
                    cache_write_token_price = (rates["cache_write"] / 1000000) * cache_write_tokens
    return {
        # "api_key": api_key, # remove for better safety
        "id": res_json['id'] if "id" in res_json else None,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "prompt_token_price": prompt_token_price,
        "completion_token_price": completion_token_price,
        "cache_read_token_price": cache_read_token_price,
        "cache_write_token_price": cache_write_token_price,
        "price_currency": price_currency,
        "service_tier": selected_service_tier,
    }


def inference_chat(chat, model, api_url, token, usage_tracking_jsonl = None, max_tokens = 2048,
                   temperature = 0.0, api_protocol = None, service_tier = None,
                   thinking = None, price_currency = "USD"):
    if token is None:
        raise ValueError("API key is required")

    if api_protocol is None:
        api_protocol = "anthropic" if "claude" in model.lower() else "openai"
    api_protocol = api_protocol.lower()
    if api_protocol not in ("openai", "anthropic"):
        raise ValueError(f"Unsupported API protocol: {api_protocol}")
    if service_tier is not None and service_tier not in ("standard", "priority"):
        raise ValueError(f"Unsupported service tier: {service_tier}")
    if thinking is not None and thinking not in ("adaptive", "disabled"):
        raise ValueError(f"Unsupported thinking mode: {thinking}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": max_tokens,
        'temperature': temperature
    }

    if service_tier is not None:
        data["service_tier"] = service_tier
    if thinking is not None:
        data["thinking"] = {"type": thinking}

    if api_protocol == "anthropic":
        if "47.88.8.18:8088" not in api_url:
            # using official api url
            headers = {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        for role, content in chat:
            if role == "system":
                assert content[0]['type'] == "text" and len(content) == 1
                data['system'] = content[0]['text']
            else:
                converted_content = []
                for item in content:
                    if item['type'] == "text":
                        converted_content.append({"type": "text", "text": item['text']})
                    elif item['type'] in ("image_url", "video_url"):
                        media_type = "image" if item['type'] == "image_url" else "video"
                        media = item[item['type']]
                        media_url = media['url']
                        if media_url.startswith("data:"):
                            metadata, encoded_data = media_url.split(",", 1)
                            source = {
                                "type": "base64",
                                "media_type": metadata.removeprefix("data:").split(";", 1)[0],
                                "data": encoded_data,
                            }
                        else:
                            source = {"type": "url", "url": media_url}
                        for option in ("detail", "fps", "max_long_side_pixel"):
                            if option in media:
                                source[option] = media[option]
                        converted_content.append({
                            "type": media_type,
                            "source": source,
                        })
                    else:
                        raise ValueError(f"Invalid content type: {item['type']}")
                data["messages"].append({"role": role, "content": converted_content})       
    else:
        for role, content in chat:
            data["messages"].append({"role": role, "content": content})

    max_retry = 5
    sleep_sec = 20

    while True:
        try:
            if api_protocol == "anthropic":
                res = requests.post(api_url, headers=headers, data=json.dumps(data))
                res_json = res.json()
                # print(res_json)
                text_blocks = [block['text'] for block in res_json['content'] if block.get('type') == 'text']
                res_content = "\n".join(text_blocks)
            else:
                res = requests.post(api_url, headers=headers, json=data)
                res_json = res.json()
                # print(res_json)
                res_content = res_json['choices'][0]['message']['content']
            if usage_tracking_jsonl:
                usage = track_usage(
                    res_json,
                    api_key=token,
                    service_tier=service_tier or "standard",
                    price_currency=price_currency,
                )
                with open(usage_tracking_jsonl, "a") as f:
                    f.write(json.dumps(usage) + "\n")
        except:
            print("Network Error:")
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break
        print(f"Sleep {sleep_sec} before retry...")
        sleep(sleep_sec)
        max_retry -= 1
        if max_retry < 0:
            print(f"Failed after {max_retry} retries...")
            return None
    
    return res_content
