## Q

In China, students can use Cloud LLM API from Tencent Cloud: https://cloud.tencent.com/document/product/1729/111007

List OpenAI like LLM (language) with API access from major players (OpenAI, Azure, AWS, Tencent Huanyuan, Alibaba, Doubao, Google, Mistral, Claude, DeepSeek, etc.) with which we can have such code: import os
from openai import OpenAI
```python
# 构造 client
client = OpenAI(
    api_key=os.environ.get("HUNYUAN_API_KEY"),  # 混元 APIKey
    base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
)

completion = client.chat.completions.create(
    model="hunyuan-turbos-latest",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test."
        }
    ],
    extra_body={
        "enable_enhancement": True,  # <- 自定义参数
    },
)
print(completion.choices[0].message.content)   
```
Tell me should I bind a credit card for getting that and how much is the free quota 


Here's a breakdown of major LLM API providers, credit card requirements, and free quotas:


## A 

**Major LLM Providers with API Access**
| Provider               | Credit Card Required? | Free Quota Details                                                                 |
|------------------------|-----------------------|-----------------------------------------------------------------------------------|
| OpenAI             | ✅ Yes                | $5 free trial credits (expire after 3 months). Pay-as-you-go after.              |
| Azure OpenAI       | ✅ Yes                | $200 Azure credits for new users. Requires separate service deployment.          |
| AWS Bedrock        | ✅ Yes                | 12-month AWS Free Tier (limited to specific services). Free trial models available. |
| Google (Gemini)    | ✅ Yes                | $300 Google Cloud credits for first 90 days.                                      |
| Tencent Hunyuan    | ✅ Yes                | Limited free tier for new Tencent Cloud accounts (specific to region/service).    |
| Alibaba Tongyi     | ✅ Yes                | Free trial via Alibaba Cloud (limited tokens/API calls).                          |
| Claude (Anthropic) | ✅ Yes                | Available via AWS Bedrock (follows AWS policies) or direct API (usage-based).     |
| Mistral            | ⚠️ Varies           | On AWS: Follows AWS rules. Direct API may require CC for production use.           |
| DeepSeek           | ✅ Likely            | Free trial available (e.g., 100M tokens). CC required for paid tier.              |
| Doubao (ByteDance) | ✅ Likely            | Limited free quota for testing (requires enterprise verification in some cases).  |

---

**Key Notes**
1. Credit Card Requirements: Most providers require a credit card for identity verification, even if offering free credits/trials.
2. Free Quotas: Typically time-limited (e.g., Google’s 90-day credits) or usage-capped (e.g., OpenAI’s $5 trial).
3. Enterprise Services: Providers like Alibaba, Tencent, or Doubao may require additional business verification for full access.
4. Regional Differences: Free tiers (e.g., Tencent Hunyuan) might vary based on your account's geographic region.

For the Tencent Hunyuan example code:  
• You’ll need a Tencent Cloud account (credit card required).  

• Check their documentation for region-specific free tiers (e.g., Singapore often has more generous trials).  


Always review the latest pricing docs, as policies change frequently!