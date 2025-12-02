"""Prompt templates shared across datasets and evaluation."""

from __future__ import annotations

from typing import Iterable, List

from rag_data_generator.tooling.tools import DEFAULT_TOOL_SPECS, ToolSpec

TOOL_DESC = """{name_for_model}: 使用 {name_for_human} 这个API交互. {description_for_model} 参数: {parameters} 格式需要是JSON对象."""

SYSTEM_PROMPT_TOOLS_BACKTRACK = '''
The user asks a question, and the assistant must solve it. The assistant should first think internally and reason through the problem, and only after reasoning should it provide the final answer.

You must follow the rules below strictly. All output must appear only inside the specified tags and no text may appear outside them.

Output format requirements:

The assistant must output exactly five sections in the following order. Each section must be wrapped in a matching pair of tags, and each tag must appear exactly once.

1. <reasoning>...</reasoning>
2. <search>...</search>
3. <summary>...</summary>
4. <backtrack>...</backtrack>
5. <answer>...</answer>

If a section has no content you must still output the tag pair with an empty body(except for search, you should not perform an empty search), for example:
<backtrack></backtrack>

Nothing may be written outside the five required tag blocks. Do not output anything resembling [ASSISTANT] or system notes or extra text.

Meaning of each tag:

<reasoning> is for internal reasoning and planning how to answer. It often occurs at the begining, or follows the observation and summary to decide whether we need a deeper search.

<search> is for tool usage or external lookup instructions. If no search is needed, skip this tag. Search query (list only the keywords, e.g. "keyword_1 keyword_2 ...")</search>.
Each search query may contain only one triple.

<summary> is an optional short summary of the developing conclusion. Leave empty if not needed.

<backtrack> is for correction or revision of earlier reasoning if necessary. Leave empty if no correction is needed.

<answer> is the final response to the user. It must contain the final answer only, with no internal reasoning or hidden thought. It must be the last block and must have a closing </answer> tag.

Critical rules:

Do not write anything in <answer> until the reasoning is complete.  
Reasoning, searching, summarizing and backtracking may continue before the final answer is determined.
Try to find the most specific answer to the question.
If the necessary information is already obtained, jump extra steps straight to answer.

Possible Structure:
<reasoning>analyze the question, plan what to query</reasoning>
<search>search for something</search>
<observation>search result</observation>
<summary>summarize the most important information</summary>
<backtrack>reflect on the previous search choices if search result is unsatisfying</backtrack>
<reasoning>maybe search for more specific answer or decide to conclude the answer</reasoning>
...
<answer>...</answer>

You have the following tools available:  
`{tool_descs}`
'''

SYSTEM_PROMPT_TOOLS_SSRL = """
用户提出一个问题，助手来解决。助手首先在脑海中思考推理过程，然后向用户提供最终答案。
1. 在思考过程中，先对已知信息进行简要分析，然后决定是否需要进一步搜索，你可以使用<reasoning> </reasoning>标签包裹你的推理分析；
2. 在思考过程中，如果你认为你需要对上文的关键信息做一些总结，你可以使用 <summary> </summary> 标签包裹你的总结提炼；
3. 在思考过程中，如果你认为上文的思考需要订正或修改，你可以使用 <backtrack> </backtrack> 标签包裹你的反思结果，用指导接下来的搜索方向；
4. <answer> 在这里写最终答案，请直接地回答问题，不需要附带多余信息 </answer>"；
5. 在思考过程中，**如果有必要，助手可以进行内心的搜索**；

answer只能在最后进行，在得出结论前可以进行多轮不同方向的、逐渐细分的检索。尽量用到多种分析能力

你有以下工具可以使用:
{tool_descs}

请按照以下格式作答：

<reasoning>
...
</reasoning>
<search>
...
</search>
<summary>
...
</summary>
<backtrack>
...
</backtrack>
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

LLM_EVAL_PROMPT = """
你是一名严格、但能识别同义表达的阅卷老师。请阅读以下信息并判断学生的选择题作答是否正确：

1. 【题目】：
{question}

2. 【正确答案】：
{expected}

3. 【学生的作答】：
{predicted}

你的任务是：
- 首先判断学生的作答是否与正确答案一致（如果含义相同也视为一致）；
- 如果学生作答正确，请只输出：Yes
- 如果学生作答错误，请只输出：No

**重要要求**：
- 不要输出引号、标点、换行、额外文字、空格或其他任何字符。
- 只输出一个单词：Yes 或 No。
"""


def build_prompt(messages: Iterable[dict]) -> str:
    return "\n".join([str(message["content"]).strip() for message in messages])


def build_system_tools(
    sys_prompt: str = SYSTEM_PROMPT_TOOLS_BACKTRACK,
    tool_specs: Iterable[ToolSpec] | None = None,
    max_results: int = 3,
) -> str:
    tool_specs = list(tool_specs or DEFAULT_TOOL_SPECS)
    descs: List[str] = []
    names: List[str] = []
    for spec in tool_specs:
        desc = TOOL_DESC.format(**spec.__dict__)
        descs.append(desc + f"\n默认最多返回 {max_results} 条结果。")
        names.append(spec.name_for_model)
    prompt = sys_prompt.format(tool_descs="\n\n".join(descs), tool_names=",".join(names))
    return prompt + f"\n请注意：每次检索只能返回前 {max_results} 条最相关的文档。"


__all__ = [
    "TOOL_DESC",
    "SYSTEM_PROMPT_TOOLS_BACKTRACK",
    "SYSTEM_PROMPT_TOOLS_SSRL",
    "LLM_EVAL_PROMPT",
    "build_prompt",
    "build_system_tools",
]
